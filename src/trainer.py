#!/usr/bin/env python3
"""
Docstring for trainer
"""

# imports
from torch.utils.data import DataLoader
import torch  
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Dict
from models import FocalLoss, ClassBalancedLoss
from huggingface_hub import snapshot_download
import pandas as pd
import optuna
import time
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


def _nll_loss_from_log_probs(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_class_balanced_loss: bool = False,
    samples_per_class=None,
    cb_beta: float = 0.9999,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute loss from log-probabilities without applying log_softmax again.

    The combined dual-stream output is a log-probability mixture distribution.
    Standard CrossEntropyLoss would apply log_softmax a second time, corrupting
    the gradients.  This function computes the correct NLL / focal-NLL /
    class-balanced-NLL directly from log-probs.
    """
    # Per-sample log-probability and probability for the target class
    log_pt = log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)  # [B]
    pt = log_pt.exp()  # [B]  (actual mixture probability of the target class)

    if use_class_balanced_loss and samples_per_class is not None:
        n = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(cb_beta, n)
        cb_w = (1.0 - cb_beta) / effective_num
        cb_w = cb_w / cb_w.sum() * len(cb_w)
        cb_w = torch.tensor(cb_w, dtype=torch.float32, device=log_probs.device)
        class_w = cb_w[labels]  # [B]
        if use_focal_loss:
            loss = -(class_w * (1 - pt) ** focal_gamma * log_pt).mean()
        else:
            loss = -(class_w * log_pt).mean()
    elif use_focal_loss:
        focal_w = (1 - pt) ** focal_gamma
        if weights is not None:
            focal_w = focal_w * weights[labels]
        loss = -(focal_w * log_pt).mean()
    else:
        loss = F.nll_loss(log_probs, labels, weight=weights)

    return loss

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
    prokaryote_idx: int = 2,
    coarse_aux_weight: float = 0.2,
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
        raw_features = batch.get('raw_features')
        if raw_features is not None:
            raw_features = raw_features.to(device)
        marker_indices = batch.get('marker_indices')
        if marker_indices is not None:
            marker_indices = marker_indices.to(device)

        # Forward pass + loss computation (shared between FP16 and FP32 paths)
        def compute_forward_and_loss():
            nonlocal total_glm_loss, total_feat_loss
            fwd_kwargs = dict(features=features, raw_features=raw_features, marker_indices=marker_indices)
            if input_ids is not None:
                fwd_kwargs['input_ids'] = input_ids.to(device)
                fwd_kwargs['attention_mask'] = attention_mask.to(device) if attention_mask is not None else None
            else:
                fwd_kwargs['sequences'] = sequences
            
            if chunk_loss_averaging:
                _logits, _loss = model(**fwd_kwargs, labels=labels)
            elif use_auxiliary_loss:
                aux_outputs = model(**fwd_kwargs, return_auxiliary_logits=True)
                if len(aux_outputs) == 4:
                    # dual_stream without coarse head: (combined, glm, feat, log_vars)
                    combined, glm_log, feat_log, log_vars = aux_outputs
                    coarse_log = None
                elif len(aux_outputs) == 5:
                    # hierarchical dual_stream with coarse head: (combined, glm, feat, coarse, log_vars)
                    combined, glm_log, feat_log, coarse_log, log_vars = aux_outputs
                else:
                    raise ValueError(f"Unexpected number of auxiliary outputs: {len(aux_outputs)}")
                if use_learnable_aux_loss and log_vars is not None:
                    l_main = _nll_loss_from_log_probs(
                        combined, labels,
                        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                        use_class_balanced_loss=use_class_balanced_loss,
                        samples_per_class=samples_per_class, cb_beta=cb_beta,
                        weights=weights.to(device) if weights is not None else None,
                    )
                    prec_main = torch.exp(-log_vars[0])
                    _loss = (prec_main * l_main) + log_vars[0]
                    if glm_log is not None:
                        l_glm = criterion(glm_log, labels)
                        prec_glm = torch.exp(-log_vars[1])
                        _loss += (prec_glm * l_glm) + log_vars[1]
                        total_glm_loss += l_glm.item()
                    if feat_log is not None:
                        l_feat = _nll_loss_from_log_probs(
                            feat_log, labels,
                            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                            use_class_balanced_loss=use_class_balanced_loss,
                            samples_per_class=samples_per_class, cb_beta=cb_beta,
                            weights=weights.to(device) if weights is not None else None,
                        )
                        prec_feat = torch.exp(-log_vars[2])
                        _loss += (prec_feat * l_feat) + log_vars[2]
                        total_feat_loss += l_feat.item()
                    if coarse_log is not None:
                        coarse_labels = (labels != prokaryote_idx).long()
                        # Plain CE — coarse head is a balanced binary task; the 5-class
                        # criterion (CB/focal with 5-class weights) would crash or give
                        # wrong per-class weights when applied to binary [B, 2] logits.
                        l_coarse = F.cross_entropy(coarse_log, coarse_labels)
                        _loss = _loss + coarse_aux_weight * l_coarse
                else:
                    _loss = _nll_loss_from_log_probs(
                        combined, labels,
                        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                        use_class_balanced_loss=use_class_balanced_loss,
                        samples_per_class=samples_per_class, cb_beta=cb_beta,
                        weights=weights.to(device) if weights is not None else None,
                    )
                    if glm_log is not None:
                        l_glm = criterion(glm_log, labels)
                        _loss = _loss + auxiliary_loss_weight * l_glm
                        total_glm_loss += l_glm.item()
                    if feat_log is not None:
                        l_feat = _nll_loss_from_log_probs(
                            feat_log, labels,
                            use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                            use_class_balanced_loss=use_class_balanced_loss,
                            samples_per_class=samples_per_class, cb_beta=cb_beta,
                            weights=weights.to(device) if weights is not None else None,
                        )
                        _loss = _loss + auxiliary_loss_weight * l_feat
                        total_feat_loss += l_feat.item()
                    if coarse_log is not None:
                        coarse_labels = (labels != prokaryote_idx).long()
                        # Plain CE — coarse head is a balanced binary task; the 5-class
                        # criterion (CB/focal with 5-class weights) would crash or give
                        # wrong per-class weights when applied to binary [B, 2] logits.
                        l_coarse = F.cross_entropy(coarse_log, coarse_labels)
                        _loss = _loss + coarse_aux_weight * l_coarse
                _logits = combined
            else:
                _logits = model(**fwd_kwargs)
                # GenomeClassifier always outputs log-probabilities; using
                # CrossEntropyLoss (which applies log_softmax internally) would
                # double-log them.  Detect the dual-stream case and use NLLLoss.
                _is_dual = (
                    hasattr(model, 'feature_integration_mode')
                    and model.feature_integration_mode in [
                        "dual_stream", "dual_stream_dynamic_gate", "stacking"
                    ]
                )
                if _is_dual:
                    _loss = _nll_loss_from_log_probs(
                        _logits, labels,
                        use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                        use_class_balanced_loss=use_class_balanced_loss,
                        samples_per_class=samples_per_class, cb_beta=cb_beta,
                        weights=weights.to(device) if weights is not None else None,
                    )
                else:
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
    compute_auprc: bool = True,
    prokaryote_idx: int = 2,
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
    all_coarse_preds = []
    all_coarse_labels_bin = []
    is_dual_stream = hasattr(model, 'feature_integration_mode') and model.feature_integration_mode in ["dual_stream", "dual_stream_dynamic_gate", "stacking"]

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
            raw_features = batch.get('raw_features')
            if raw_features is not None:
                raw_features = raw_features.to(device)
            marker_indices = batch.get('marker_indices')
            if marker_indices is not None:
                marker_indices = marker_indices.to(device)

            fwd_kwargs = dict(features=features, raw_features=raw_features, marker_indices=marker_indices)
            if input_ids is not None:
                fwd_kwargs['input_ids'] = input_ids.to(device)
                fwd_kwargs['attention_mask'] = attention_mask.to(device) if attention_mask is not None else None
            else:
                fwd_kwargs['sequences'] = sequences

            if is_dual_stream:
                aux = model(**fwd_kwargs, return_auxiliary_logits=True)
                if len(aux) == 4:
                    logits, glm_logits, feat_logits, _ = aux
                    coarse_logits = None
                else:  # 5-tuple: coarse head present
                    logits, glm_logits, feat_logits, coarse_logits, _ = aux
                if glm_logits is not None:
                    all_glm_preds.extend(torch.argmax(glm_logits, dim=-1).cpu().numpy())
                if feat_logits is not None:
                    all_feat_preds.extend(torch.argmax(feat_logits, dim=-1).cpu().numpy())
                if coarse_logits is not None:
                    all_coarse_preds.extend(torch.argmax(coarse_logits, dim=-1).cpu().numpy())
                    all_coarse_labels_bin.extend(
                        (labels != prokaryote_idx).long().cpu().numpy()
                    )
            else:
                logits = model(**fwd_kwargs)

            # Extract gate alpha (sigmoid scalar per sample)
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
                    all_alphas.extend([float(a) for a in batch_alphas])
            
            if is_dual_stream:
                loss = _nll_loss_from_log_probs(
                    logits, labels,
                    use_focal_loss=use_focal_loss, focal_gamma=focal_gamma,
                    use_class_balanced_loss=use_class_balanced_loss,
                    samples_per_class=samples_per_class, cb_beta=cb_beta,
                    weights=weights.to(device) if weights is not None else None,
                )
            else:
                loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.exp(logits) if is_dual_stream else torch.softmax(logits, dim=-1)
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

    all_probs_np = np.array(all_probs)
    n_classes = all_probs_np.shape[1]
    if compute_auprc:
        val_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
        if val_true_bin.shape[1] == 1:  # binary edge case
            val_true_bin = np.hstack((1 - val_true_bin, val_true_bin))
        # Only score classes that are actually present — per-clade holdout loaders
        # contain a single family, leaving most columns all-zero and causing
        # "No positive class found" warnings and inflated AUPRC values.
        present_classes = sorted(set(all_labels))
        if len(present_classes) >= 2:
            macro_auprc = average_precision_score(
                val_true_bin[:, present_classes],
                all_probs_np[:, present_classes],
                average='macro',
            )
        else:
            macro_auprc = 0.0
    else:
        macro_auprc = 0.0

    glm_accuracy = None
    xgb_accuracy = None
    coarse_accuracy = None
    if all_glm_preds:
        glm_accuracy = accuracy_score(all_labels, all_glm_preds)
    if all_feat_preds:
        xgb_accuracy = accuracy_score(all_labels, all_feat_preds)
    if all_coarse_preds:
        coarse_accuracy = accuracy_score(all_coarse_labels_bin, all_coarse_preds)

    return avg_loss, accuracy, macro_f1, macro_auprc, weighted_f1, min_class_f1, all_preds, all_labels, all_accessions, all_probs, glm_accuracy, xgb_accuracy, all_alphas, coarse_accuracy

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
    resume_best_macro_auprc: float = 0.0,
    resume_best_worst_holdout_acc: float = 0.0,
    resume_best_min_class_f1: float = 0.0,
    resume_patience_counter: int = 0,
    use_auxiliary_loss: bool = False,
    auxiliary_loss_weight: float = 0.3,
    use_learnable_aux_loss: bool = False,
    challenge_loaders: Optional[Dict[str, DataLoader]] = None,
    trial=None,
    prokaryote_idx: int = 2,
    coarse_aux_weight: float = 0.2,
) -> Tuple[nn.Module, dict]:
    """Main training loop."""
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    if start_epoch > 1:
        print(f"Resuming training from epoch {start_epoch}")

    best_val_acc = resume_best_val_acc
    best_val_loss = resume_best_val_loss
    best_macro_auprc = resume_best_macro_auprc
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
            'val_macro_auprc': [],
            'val_weighted_f1': [],
            'val_worst_holdout_acc': [],
        }

    for epoch in range(start_epoch, epochs + 1):

        # per epoch tracking
        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

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
            prokaryote_idx=prokaryote_idx,
            coarse_aux_weight=coarse_aux_weight,
        )

        # report on the epoch stats 
        epoch_duration = time.time() - epoch_start_time
        epoch_peak_vram = 0 
        if torch.cuda.is_available():
            epoch_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # in GB

        # append to history 
        history.setdefault('epoch_times', []).append(epoch_duration)
        history.setdefault('epoch_peak_vram', []).append(epoch_peak_vram)
        
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
        
        val_loss, val_acc, macro_f1, macro_auprc, weighted_f1, min_class_f1, val_preds, val_true, val_accs_list, val_probs, glm_acc, xgb_acc, val_alphas, coarse_acc = evaluate(
            model, dataloader=val_loader, device=device, epoch=epoch, weights=weights, use_focal_loss=use_focal_loss, focal_gamma=focal_loss_gamma,
            use_class_balanced_loss=use_class_balanced_loss, samples_per_class=samples_per_class, cb_beta=cb_beta,
            prokaryote_idx=prokaryote_idx,
        )
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_macro_f1'].append(macro_f1)
        if 'val_macro_auprc' not in history:
            history['val_macro_auprc'] = []
        history['val_macro_auprc'].append(macro_auprc)
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
                _, c_acc, _, _, _, _, _, _, _, _, _, _, _, _ = evaluate(
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
                    compute_auprc=False,
                    prokaryote_idx=prokaryote_idx,
                )
                challenge_accuracies[family_name] = c_acc
                print(f"    - {family_name}: {c_acc:.4f}")

            if challenge_accuracies:
                epoch_worst_challenge_acc = min(challenge_accuracies.values())
                acc_values = [float(acc) for acc in challenge_accuracies.values() if acc is not None]
                if acc_values:
                    # Smoothed harmonic mean to prevent a single 0.0 from flatlining the gradient
                    smoothed_accs = [acc + 0.01 for acc in acc_values]
                    hmean = len(smoothed_accs) / sum(1.0 / acc for acc in smoothed_accs)
                    epoch_holdout_hmean_acc = max(0.0, hmean - 0.01)
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
        
        # --- Stream weight: mean alpha from dynamic gate ---
        epoch_stream_weight = None
        if val_alphas and len(val_alphas) > 0:
            epoch_stream_weight = sum(val_alphas) / len(val_alphas)
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
        if xgb_acc is not None:
            if 'xgb_accuracy' not in history:
                history['xgb_accuracy'] = []
            history['xgb_accuracy'].append(xgb_acc)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss:      {train_loss:.4f}")
        if use_auxiliary_loss:
            print(f"  Train GLM Loss:  {train_glm_loss:.4f}")
            print(f"  Train Feat Loss: {train_feat_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Val Acc:         {val_acc:.4f}")
        print(f"  Macro F1:        {macro_f1:.4f}")
        print(f"  Macro AUPRC:     {macro_auprc:.4f}")
        print(f"  Min-Class F1:    {min_class_f1:.4f}")
        print(f"  Weighted F1:     {weighted_f1:.4f}")
        if epoch_worst_challenge_acc is not None:
            print(f"  Worst Holdout Acc: {epoch_worst_challenge_acc:.4f}")
        if epoch_holdout_hmean_acc is not None:
            print(f"  Harmonic Mean Holdout Acc: {epoch_holdout_hmean_acc:.4f}")
        
        # --- NEW EXPLICIT LOGGING FOR STREAM WEIGHTS ---
        if stream_weight is not None:
            print(f"  Avg Gate Weight: GLM={stream_weight:.1%}, XGBoost={1-stream_weight:.1%}")

        if getattr(model, "loss_log_vars", None) is not None:
            precisions = torch.exp(-model.loss_log_vars).detach().cpu().numpy()
            print(f"  Learnable Aux Weights (Uncertainty Precision):")
            print(f"    - Combined Output: {precisions[0]:.4f}")
            print(f"    - GLM Output:      {precisions[1]:.4f}")
            print(f"    - Feature Output:  {precisions[2]:.4f}")
            
        if glm_acc is not None or xgb_acc is not None:
            glm_acc_str = f"{glm_acc:.4f}" if glm_acc is not None else "N/A"
            xgb_acc_str = f"{xgb_acc:.4f}" if xgb_acc is not None else "N/A"
            print(f"  Per-Stream Acc:  GLM={glm_acc_str}, XGBoost={xgb_acc_str}")
        if coarse_acc is not None:
            if 'coarse_accuracy' not in history:
                history['coarse_accuracy'] = []
            history['coarse_accuracy'].append(coarse_acc)
            print(f"  Coarse Acc (prok/euk): {coarse_acc:.4f}")
        
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
            class_alphas = {k: [] for k in idx_to_label.keys()}
            for true_class, alpha_val in zip(val_true, val_alphas):
                class_alphas[true_class].append(alpha_val)

            print(f"\n  Per-Class Average Gate Weights (α = GLM weight):")
            for class_idx in sorted(idx_to_label.keys()):
                class_name = idx_to_label[class_idx]
                alphas_list = class_alphas[class_idx]
                if len(alphas_list) > 0:
                    mean_alpha = sum(alphas_list) / len(alphas_list)
                    print(f"    - {class_name:>27}: {mean_alpha:.4f} (GLM={mean_alpha:>5.1%}, XGBoost={1-mean_alpha:>5.1%})")

        
        # -----------------------------------------------------------------
        # DUAL CHECKPOINTING & EARLY STOPPING
        # -----------------------------------------------------------------
        is_best_macro = False
        is_best_holdout = False

        min_delta = 0.001

        if macro_auprc > (best_macro_auprc + min_delta) or (epoch == start_epoch and best_macro_auprc == 0.0):
            best_macro_auprc = macro_auprc
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
            patience_msg = f"No objective improved (best macro_auprc={best_macro_auprc:.4f}, min_class_f1={best_min_class_f1_for_stopping:.4f}, holdout_hmean={best_worst_holdout_acc:.4f})."
        else:
            primary_improved = is_best_macro
            patience_msg = f"No objective improved (best macro_auprc={best_macro_auprc:.4f}, min_class_f1={best_min_class_f1_for_stopping:.4f})."

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
                'best_macro_auprc': best_macro_auprc,
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
            save_best_checkpoint("best_macro_auprc_model", "macro AUPRC", macro_auprc)
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
                trial.report(macro_auprc, epoch) 
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
            'best_macro_auprc': best_macro_auprc,
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
