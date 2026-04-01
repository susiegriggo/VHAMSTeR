#!/usr/bin/env python3
"""
Docstring for trainer
"""

# imports
from torch.utils.data import DataLoader
import torch  
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Dict
try:
    from .models import FocalLoss, ClassBalancedLoss
except ImportError:
    from models import FocalLoss, ClassBalancedLoss
from huggingface_hub import snapshot_download
import pandas as pd
import optuna
import json
import shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# mixed precision support
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    weights: torch.Tensor = None,
    use_fp16: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    chunk_loss_averaging: bool = False,
    gradient_accumulation_steps: int = 1,
    use_class_balanced_loss: bool = False,
    samples_per_class: list = None,
    cb_beta: float = 0.9999,
    use_auxiliary_loss: bool = False,
    auxiliary_loss_weight: float = 0.3,
    use_learnable_aux_loss: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_glm_loss = 0 # log the loss specific to the glm stream
    total_feat_loss = 0 # log the loss specicfic to the feature stream
    accumulation_counter = 0
    
    # Create loss function (class-balanced, focal, or standard cross-entropy)
    if use_class_balanced_loss and samples_per_class is not None:
        loss_type = 'focal' if use_focal_loss else 'softmax'
        criterion = ClassBalancedLoss(
            beta=cb_beta,
            samples_per_class=samples_per_class,
            loss_type=loss_type,
            gamma=focal_gamma
        )
    elif use_focal_loss:
        criterion = FocalLoss(
            alpha=weights.to(device) if weights is not None else None,
            gamma=focal_gamma
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    
    # Setup mixed precision if requested
    scaler = GradScaler(device) if use_fp16 and AMP_AVAILABLE and device == "cuda" else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Get batch data — accept either pre-tokenized (input_ids) or raw sequences
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        sequences = batch.get('sequences')  # legacy fallback
        labels = batch['labels'].to(device)
        features = batch.get('features')
        if features is not None:
            features = features.to(device)
        
        # Forward pass + loss computation (shared between FP16 and FP32 paths)
        def compute_forward_and_loss():
            nonlocal total_glm_loss, total_feat_loss
            fwd_kwargs = dict(features=features)
            if input_ids is not None:
                fwd_kwargs['input_ids'] = input_ids.to(device)
                fwd_kwargs['attention_mask'] = attention_mask.to(device) if attention_mask is not None else None
            else:
                fwd_kwargs['sequences'] = sequences
            
            if chunk_loss_averaging:
                _logits, _loss = model(**fwd_kwargs, labels=labels)
            elif use_auxiliary_loss:
                combined, glm_log, feat_log, log_vars = model(
                    **fwd_kwargs, return_auxiliary_logits=True
                )
                if use_learnable_aux_loss and log_vars is not None:
                    l_main = criterion(combined, labels)
                    prec_main = torch.exp(-log_vars[0])
                    _loss = (prec_main * l_main) + log_vars[0]
                    if glm_log is not None:
                        l_glm = criterion(glm_log, labels)
                        prec_glm = torch.exp(-log_vars[1])
                        _loss += (prec_glm * l_glm) + log_vars[1]
                        total_glm_loss += l_glm.item()
                    if feat_log is not None:
                        l_feat = criterion(feat_log, labels)
                        prec_feat = torch.exp(-log_vars[2])
                        _loss += (prec_feat * l_feat) + log_vars[2]
                        total_feat_loss += l_feat.item()
                else:
                    _loss = criterion(combined, labels)
                    if glm_log is not None:
                        l_glm = criterion(glm_log, labels)
                        _loss = _loss + auxiliary_loss_weight * l_glm
                        total_glm_loss += l_glm.item()
                    if feat_log is not None:
                        l_feat = criterion(feat_log, labels)
                        _loss = _loss + auxiliary_loss_weight * l_feat
                        total_feat_loss += l_feat.item()
                _logits = combined
            else:
                _logits = model(**fwd_kwargs)
                _loss = criterion(_logits, labels)
            return _logits, _loss
        
        # FP16 vs FP32 path differs only in autocast wrapper and scaler usage
        if scaler is not None:
            with autocast(device):
                logits, loss = compute_forward_and_loss()
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulation_counter += 1
            if accumulation_counter % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            logits, loss = compute_forward_and_loss()
            loss = loss / gradient_accumulation_steps
            loss.backward()
            accumulation_counter += 1
            if accumulation_counter % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
    
    if accumulation_counter % gradient_accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer) # <--- UNSCALE BEFORE CLIPPING
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)

    if use_auxiliary_loss:
        avg_glm_loss = total_glm_loss / len(dataloader)
        avg_feat_loss = total_feat_loss / len(dataloader)
    else: 
        avg_glm_loss = None 
        avg_feat_loss = None 

    return avg_loss, avg_glm_loss, avg_feat_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int = None,
    weights: torch.Tensor = None,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_balanced_loss: bool = False,
    samples_per_class: list = None,
    cb_beta: float = 0.9999,
) -> Tuple[float, float, float, float, List[int], List[int], List[str], List[List[float]], Optional[float], Optional[float]]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_accessions = []
    all_probs = []  
    all_alphas = [] # For storing dynamic gate weights if applicable
    
    if use_class_balanced_loss and samples_per_class is not None:
        loss_type = 'focal' if use_focal_loss else 'softmax'
        criterion = ClassBalancedLoss(
            beta=cb_beta,
            samples_per_class=samples_per_class,
            loss_type=loss_type,
            gamma=focal_gamma
        )
    elif use_focal_loss:
        criterion = FocalLoss(
            alpha=weights.to(device) if weights is not None else None,
            gamma=focal_gamma
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    
    desc = f"Epoch {epoch} [Val]" if epoch is not None else "Evaluating"
    pbar = tqdm(dataloader, desc=desc)
    
    all_glm_preds = []
    all_feat_preds = []
    is_dual_stream = hasattr(model, 'feature_integration_mode') and model.feature_integration_mode in ["dual_stream", "dual_stream_dynamic_gate"]
    
    with torch.no_grad():
        for batch in pbar:
            # Accept either pre-tokenized (input_ids) or raw sequences
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            sequences = batch.get('sequences')  # legacy fallback
            labels = batch['labels'].to(device)
            accessions = batch['accession']
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            fwd_kwargs = dict(features=features)
            if input_ids is not None:
                fwd_kwargs['input_ids'] = input_ids.to(device)
                fwd_kwargs['attention_mask'] = attention_mask.to(device) if attention_mask is not None else None
            else:
                fwd_kwargs['sequences'] = sequences
            
            if is_dual_stream:
                logits, glm_logits, feat_logits, _ = model(
                    **fwd_kwargs, return_auxiliary_logits=True
                )
                if glm_logits is not None:
                    all_glm_preds.extend(torch.argmax(glm_logits, dim=-1).cpu().numpy())
                if feat_logits is not None:
                    all_feat_preds.extend(torch.argmax(feat_logits, dim=-1).cpu().numpy())
            else:
                logits = model(**fwd_kwargs)

            # If the model has a dynamic gate, extract alpha values safely across scalar/array types.
            if hasattr(model, "_last_alpha_batch") and model._last_alpha_batch is not None:
                batch_alphas = model._last_alpha_batch
                if isinstance(batch_alphas, (int, float)):
                    all_alphas.append(float(batch_alphas))
                elif hasattr(batch_alphas, "tolist"):
                    alphas_list = batch_alphas.tolist()
                    if isinstance(alphas_list, list):
                        all_alphas.extend(alphas_list)
                    else:
                        all_alphas.append(float(alphas_list))
                elif isinstance(batch_alphas, (list, tuple)):
                    all_alphas.extend([float(alpha) for alpha in batch_alphas])
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=-1)
            all_probs.extend(probs.cpu().numpy())
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
            all_accessions.extend(accessions)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    min_class_f1 = float(per_class_f1.min()) if len(per_class_f1) > 0 else 0.0
    
    glm_accuracy = None
    feat_accuracy = None
    if all_glm_preds:
        glm_accuracy = accuracy_score(all_labels, all_glm_preds)
    if all_feat_preds:
        feat_accuracy = accuracy_score(all_labels, all_feat_preds)
    
    return avg_loss, accuracy, macro_f1, weighted_f1, min_class_f1, all_preds, all_labels, all_accessions, all_probs, glm_accuracy, feat_accuracy, all_alphas

def train_model(
    optimizer,
    scheduler,
    epochs,
    model,
    train_loader,
    val_loader,
    device,
    weights: torch.Tensor = None,
    use_fp16: bool = False,
    use_focal_loss: bool = False,
    focal_loss_gamma: float = 2.0,
    chunk_loss_averaging: bool = False,
    gradient_accumulation_steps: int = 1,
    idx_to_label=None,
    args=None,
    tokenizer=None,
    use_class_balanced_loss: bool = False,
    samples_per_class: list = None,
    cb_beta: float = 0.9999,
    start_epoch: int = 1,
    resume_history: dict = None,
    resume_best_val_acc: float = 0.0,
    resume_best_val_loss: float = float('inf'),
    resume_best_macro_f1: float = 0.0,
    resume_best_worst_holdout_acc: float = 0.0,
    resume_best_min_class_f1: float = 0.0,
    resume_patience_counter: int = 0,
    use_auxiliary_loss: bool = False,
    auxiliary_loss_weight: float = 0.3,
    use_learnable_aux_loss: bool = False,
    challenge_loaders: Optional[Dict[str, DataLoader]] = None,
    trial=None,
) -> Tuple[nn.Module, dict]:
    """Main training loop."""
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    if start_epoch > 1:
        print(f"Resuming training from epoch {start_epoch}")

    best_val_acc = resume_best_val_acc
    best_val_loss = resume_best_val_loss
    best_macro_f1 = resume_best_macro_f1
    best_worst_holdout_acc = resume_best_worst_holdout_acc
    best_min_class_f1_for_stopping = resume_best_min_class_f1
    patience_counter = resume_patience_counter
    
    if resume_history is not None:
        history = resume_history
    else:
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_macro_f1': [],
            'val_weighted_f1': [],
            'val_worst_holdout_acc': [],
        }

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        # Update sampler epoch so dynamic resamplers produce different subsets each epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, train_glm_loss, train_feat_loss = train_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer, scheduler=scheduler, device=device, epoch=epoch, weights=weights,
            use_fp16=use_fp16, use_focal_loss=use_focal_loss, focal_gamma=focal_loss_gamma, chunk_loss_averaging=chunk_loss_averaging,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_class_balanced_loss=use_class_balanced_loss,
            samples_per_class=samples_per_class,
            cb_beta=cb_beta,
            use_auxiliary_loss=use_auxiliary_loss,
            auxiliary_loss_weight=auxiliary_loss_weight,
            use_learnable_aux_loss=use_learnable_aux_loss,
        )
        history['train_loss'].append(train_loss)
        if use_auxiliary_loss:
            #history['train_glm_loss'].append(train_glm_loss)
            #history['train_feat_loss'].append(train_feat_loss)

            # Log auxiliary losses separately for better visibility, but only if they are being used
            if 'train_glm_loss' not in history: 
                history['train_glm_loss'] = [] 
            history['train_glm_loss'].append(train_glm_loss)

            if 'train_feat_loss' not in history: 
                history['train_feat_loss'] = [] 
            history['train_feat_loss'].append(train_feat_loss)

        if hasattr(train_loader.sampler, "get_category_counts"):
            try:
                counts = train_loader.sampler.get_category_counts()
                clean_counts = {int(k): int(v) for k, v in counts.items()}
                print(f"Epoch {epoch} - Sampled class counts in training set: {clean_counts}")
            except Exception:
                print(f"Epoch {epoch} - Sampler does not support get_category_counts().")
        else:
            print(f"Epoch {epoch} - Sampler does not support get_category_counts().")
        
        val_loss, val_acc, macro_f1, weighted_f1, min_class_f1, val_preds, val_true, val_accs_list, val_probs, glm_acc, feat_acc, val_alphas = evaluate(
            model, dataloader=val_loader, device=device, epoch=epoch, weights=weights, use_focal_loss=use_focal_loss, focal_gamma=focal_loss_gamma,
            use_class_balanced_loss=use_class_balanced_loss, samples_per_class=samples_per_class, cb_beta=cb_beta
        )
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_macro_f1'].append(macro_f1)      
        history['val_weighted_f1'].append(weighted_f1)
        if 'val_min_class_f1' not in history:
            history['val_min_class_f1'] = []
        history['val_min_class_f1'].append(min_class_f1)

        epoch_worst_challenge_acc = None
        epoch_holdout_hmean_acc = None
        challenge_accuracies = {}
        if challenge_loaders:
            print("\n  Holdout Family Evaluation:")
            for family_name, c_loader in challenge_loaders.items():
                _, c_acc, _, _, _, _, _, _, _, _, _, _ = evaluate(
                    model,
                    dataloader=c_loader,
                    device=device,
                    epoch=epoch,
                    weights=weights,
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_loss_gamma,
                    use_class_balanced_loss=use_class_balanced_loss,
                    samples_per_class=samples_per_class,
                    cb_beta=cb_beta,
                )
                challenge_accuracies[family_name] = c_acc
                print(f"    - {family_name}: {c_acc:.4f}")

            if challenge_accuracies:
                epoch_worst_challenge_acc = min(challenge_accuracies.values())
                acc_values = [float(acc) for acc in challenge_accuracies.values() if acc is not None]
                if acc_values:
                    # Safe harmonic mean for holdout robustness aggregation.
                    if any(acc <= 0.0 for acc in acc_values):
                        epoch_holdout_hmean_acc = 0.0
                    else:
                        epoch_holdout_hmean_acc = len(acc_values) / sum(1.0 / acc for acc in acc_values)
                print(f"    -> Worst-Case Holdout Acc: {epoch_worst_challenge_acc:.4f}")
                if epoch_holdout_hmean_acc is not None:
                    print(f"    -> Harmonic-Mean Holdout Acc: {epoch_holdout_hmean_acc:.4f}")

        if challenge_loaders is not None:
            if 'worst_challenge_acc' not in history:
                history['worst_challenge_acc'] = []
            history['worst_challenge_acc'].append(epoch_worst_challenge_acc)

            if 'holdout_hmean_acc' not in history:
                history['holdout_hmean_acc'] = []
            history['holdout_hmean_acc'].append(epoch_holdout_hmean_acc)

            if 'challenge_accuracy_by_clade' not in history:
                history['challenge_accuracy_by_clade'] = []
            history['challenge_accuracy_by_clade'].append(challenge_accuracies)

        if 'val_worst_holdout_acc' in history:
            history['val_worst_holdout_acc'].append(epoch_worst_challenge_acc)
        if 'val_holdout_hmean_acc' in history:
            history['val_holdout_hmean_acc'].append(epoch_holdout_hmean_acc)
        
        # --- Stream weight: use full-epoch mean for dynamic gate, scalar param for static ---
        epoch_stream_weight = None
        
        # Scenario A: Dynamic Gate (use the full list of alphas collected during evaluate)
        if val_alphas and len(val_alphas) > 0:
            epoch_stream_weight = sum(val_alphas) / len(val_alphas)
            
        # Scenario B: Static Gate (use the global parameter)
        elif hasattr(model, 'get_stream_weight'):
            epoch_stream_weight = model.get_stream_weight()

        if epoch_stream_weight is not None:
            if 'stream_weight' not in history:
                history['stream_weight'] = []
            history['stream_weight'].append(epoch_stream_weight)
        
        # Keep a local reference for epoch summary logging below
        stream_weight = epoch_stream_weight
        
        if glm_acc is not None:
            if 'glm_accuracy' not in history:
                history['glm_accuracy'] = []
            history['glm_accuracy'].append(glm_acc)
        if feat_acc is not None:
            if 'feature_accuracy' not in history:
                history['feature_accuracy'] = []
            history['feature_accuracy'].append(feat_acc)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss:      {train_loss:.4f}")
        if use_auxiliary_loss:
            print(f"  Train GLM Loss:  {train_glm_loss:.4f}")
            print(f"  Train Feat Loss: {train_feat_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Val Acc:         {val_acc:.4f}")
        print(f"  Macro F1:        {macro_f1:.4f}")
        print(f"  Min-Class F1:    {min_class_f1:.4f}")
        print(f"  Weighted F1:     {weighted_f1:.4f}")
        if epoch_worst_challenge_acc is not None:
            print(f"  Worst Holdout Acc: {epoch_worst_challenge_acc:.4f}")
        if epoch_holdout_hmean_acc is not None:
            print(f"  Harmonic Mean Holdout Acc: {epoch_holdout_hmean_acc:.4f}")
        
        # --- NEW EXPLICIT LOGGING FOR STREAM WEIGHTS & UNCERTAINTY ---
        if stream_weight is not None:
            if getattr(model, "feature_integration_mode", "") == "dual_stream_dynamic_gate":
                print(f"  Avg Dynamic Gate Weight: {stream_weight:.4f} (Avg GLM={stream_weight:.1%}, Avg Features={1-stream_weight:.1%})")
            else:
                print(f"  Global Stream Weight:    {stream_weight:.4f} (GLM={stream_weight:.1%}, Features={1-stream_weight:.1%})")
                
        if getattr(model, "loss_log_vars", None) is not None:
            precisions = torch.exp(-model.loss_log_vars).detach().cpu().numpy()
            print(f"  Learnable Aux Weights (Uncertainty Precision):")
            print(f"    - Combined Output: {precisions[0]:.4f}")
            print(f"    - GLM Output:      {precisions[1]:.4f}")
            print(f"    - Feature Output:  {precisions[2]:.4f}")
            
        if glm_acc is not None or feat_acc is not None:
              glm_acc_str = f"{glm_acc:.4f}" if glm_acc is not None else "N/A"
              feat_acc_str = f"{feat_acc:.4f}" if feat_acc is not None else "N/A"
              print(f"  Per-Stream Acc:  GLM={glm_acc_str}, Features={feat_acc_str}")
        
        print("\nPer-class Validation Metrics:")
        report = classification_report(
            val_true, val_preds,
            target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
            digits=3,
            output_dict=False,
            zero_division=0,
        )
        print(report)

        # print the results of the dynamic gating 
        if val_alphas and len(val_alphas) == len(val_true):
            print(f"\n  Per-Class Average Dynamic Gate Weights (α):")
            
            # Create a dictionary to hold the alphas for each class
            class_alphas = {k: [] for k in idx_to_label.keys()}
            
            # Group every alpha by its true biological label
            for true_class, alpha_val in zip(val_true, val_alphas):
                class_alphas[true_class].append(alpha_val)
                
            # Calculate and print the mean for each class
            for class_idx in sorted(idx_to_label.keys()):
                class_name = idx_to_label[class_idx]
                alphas_list = class_alphas[class_idx]
                
                if len(alphas_list) > 0:
                    mean_alpha = sum(alphas_list) / len(alphas_list)
                    print(f"    - {class_name:>27}: {mean_alpha:.4f} (GLM={mean_alpha:>5.1%}, Features={1-mean_alpha:>5.1%})")

        
        # -----------------------------------------------------------------
        # DUAL CHECKPOINTING & EARLY STOPPING
        # -----------------------------------------------------------------
        is_best_macro = False
        is_best_holdout = False

        if macro_f1 > best_macro_f1 or (epoch == start_epoch and best_macro_f1 == 0.0):
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            best_val_acc = val_acc
            is_best_macro = True

        if epoch_holdout_hmean_acc is not None:
            if epoch_holdout_hmean_acc > best_worst_holdout_acc or (epoch == start_epoch and best_worst_holdout_acc == 0.0):
                best_worst_holdout_acc = epoch_holdout_hmean_acc
                is_best_holdout = True

        # Track min-class F1 improvement for OR-based patience.
        is_best_min_class = min_class_f1 > best_min_class_f1_for_stopping or (epoch == start_epoch and best_min_class_f1_for_stopping == 0.0)
        if is_best_min_class:
            best_min_class_f1_for_stopping = min_class_f1

        # Determine early stopping logic: reset counter if ANY objective improves.
        # This prevents early stopping from firing while minority-class F1 is still climbing,
        # which can happen when imbalance strategies are active.
        if challenge_loaders:
            primary_improved = is_best_holdout
            patience_msg = f"No objective improved (best macro_f1={best_macro_f1:.4f}, min_class_f1={best_min_class_f1_for_stopping:.4f}, holdout_hmean={best_worst_holdout_acc:.4f})."
        else:
            primary_improved = is_best_macro
            patience_msg = f"No objective improved (best macro_f1={best_macro_f1:.4f}, min_class_f1={best_min_class_f1_for_stopping:.4f})."

        any_objective_improved = primary_improved or is_best_min_class
        next_patience_counter = 0 if any_objective_improved else (patience_counter + 1)

        def save_best_checkpoint(save_dir_name, metric_desc, metric_val):
            print(f"  ✓ New best {metric_desc} ({metric_val:.4f})! Saving to {save_dir_name}...")
            model_save_path = args.output_dir / save_dir_name
            model_save_path.mkdir(exist_ok=True)

            if hasattr(model.base_model, "save_pretrained"):
                model.base_model.save_pretrained(model_save_path)

            tokenizer.save_pretrained(model_save_path)
            torch.save(model.state_dict(), model_save_path / "classifier_head.pt")

            if args.model_type == "bibert":
                try:
                    repo_path = snapshot_download(repo_id="Lancelot53/birnabert-2ep")
                    for py_file in Path(repo_path).glob("*.py"):
                        dst_path = model_save_path / py_file.name
                        if not dst_path.exists() or py_file.stat().st_size != dst_path.stat().st_size:
                            shutil.copy(py_file, dst_path)
                except Exception:
                    pass

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'macro_f1': macro_f1,
                'worst_challenge_acc': epoch_worst_challenge_acc,
                'holdout_hmean_acc': epoch_holdout_hmean_acc,
                'challenge_accuracies': challenge_accuracies,
                'best_macro_f1': best_macro_f1,
                'best_worst_holdout_acc': best_worst_holdout_acc,
                'best_min_class_f1': best_min_class_f1_for_stopping,
                'patience_counter': next_patience_counter,
            }, model_save_path / "training_state.pt")

            pred_dict = {
                'accession': val_accs_list,
                'true_label': [idx_to_label[l] for l in val_true],
                'pred_label': [idx_to_label[p] for p in val_preds],
                'correct': [t == p for t, p in zip(val_true, val_preds)],
            }
            val_probs_np = np.array(val_probs)
            for idx in sorted(idx_to_label.keys()):
                pred_dict[f"prob_{idx_to_label[idx]}"] = val_probs_np[:, idx]

            pd.DataFrame(pred_dict).to_csv(model_save_path / "best_validation_probabilities.csv", index=False)

            report = classification_report(
                val_true,
                val_preds,
                target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
                digits=4,
                zero_division=0,
            )
            with open(model_save_path / "classification_report.txt", 'w') as f:
                f.write(f"Best {metric_desc}: {metric_val:.4f}\n")
                f.write(f"Epoch: {epoch}\n\n")
                f.write(report)

            cm_df = pd.DataFrame(
                confusion_matrix(val_true, val_preds),
                index=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
                columns=[idx_to_label[i] for i in sorted(idx_to_label.keys())],
            )
            cm_df.to_csv(model_save_path / "confusion_matrix.csv")

        if is_best_macro:
            save_best_checkpoint("best_macro_f1_model", "macro F1", macro_f1)
        if is_best_holdout:
            save_best_checkpoint("best_holdout_model", "Holdout Harmonic-Mean Acc", epoch_holdout_hmean_acc)

        if any_objective_improved:
            patience_counter = 0
        else:
            patience_counter = next_patience_counter
            print(f"  {patience_msg} Patience: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

        # --- Optuna mid-training pruning ---
        if trial is not None: 
            try: 
                trial.report(macro_f1,epoch) 
                if trial.should_prune():
                    print(f"\nOptuna pruned trial at epoch {epoch} (macro_f1={macro_f1:.4f}).")
                    raise optuna.TrialPruned()
            except NotImplementedError:
                # silently skip pruning if running if multi-objective optimization
                pass
            except Exception: 
                # Be safe 
                pass    
                          
        
        with open(args.output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
            
        # --- Per-epoch checkpoint: save only latest to avoid disk bloat ---
        # Only keep the latest checkpoint (+ best model directories). Old epoch dirs are removed.
        epoch_save_path = args.output_dir / f"epoch_{epoch}"
        epoch_save_path.mkdir(exist_ok=True)
        
        # Save the training state for resuming
        epoch_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_macro_f1': best_macro_f1,
            'best_worst_holdout_acc': best_worst_holdout_acc,
            'best_min_class_f1': best_min_class_f1_for_stopping,
            'patience_counter': patience_counter,
            'worst_challenge_acc': epoch_worst_challenge_acc,
            'holdout_hmean_acc': epoch_holdout_hmean_acc,
            'challenge_accuracies': challenge_accuracies,
            'history': history,
        }
        torch.save(epoch_state, epoch_save_path / "training_state.pt")
        print(f"  Saved checkpoint for epoch {epoch} to {epoch_save_path}")
        
        # Keep latest_checkpoint.pt updated for easy resuming if your job crashes
        checkpoint_path = args.output_dir / "latest_checkpoint.pt"
        torch.save(epoch_state, checkpoint_path)
        
        # Clean up previous epoch directory to prevent storage hemorrhage
        prev_epoch_path = args.output_dir / f"epoch_{epoch - 1}"
        if prev_epoch_path.exists():
            shutil.rmtree(prev_epoch_path, ignore_errors=True)
            print(f"  Cleaned up old checkpoint: {prev_epoch_path}")
        
        if device == "cuda":
            torch.cuda.empty_cache()

    return model, history