#!/usr/bin/env python3
"""
Docstring for predictor
"""

# imports
import torch 
from tqdm.auto import tqdm


def predict_batch(model, batch, device, use_fp16=False):
    """Single batch prediction with optimized tensor handling."""
    inputs = {}
    # Only pass keys relevant for inference - exclude accession and labels
    exclude_keys = {'accession', 'labels'}
    for key, val in batch.items():
        if key in exclude_keys:
            continue
        # Only move tensors to device, use non_blocking for async transfer
        if hasattr(val, 'to'):
            inputs[key] = val.to(device, non_blocking=True)
        else:
            inputs[key] = val
    
    # Use automatic mixed precision for faster inference
    if use_fp16 and device == 'cuda':
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
    
    predictions = torch.exp(outputs).cpu().numpy()
    return predictions


def predict_loader(model, dataloader, device, use_fp16=False, show_progress=True): 
    """
    Make predictions for all batches in a DataLoader
    
    Args:
        model: The model to use for predictions
        dataloader: DataLoader with batches to predict
        device: Device to run inference on
        use_fp16: Use FP16 mixed precision for faster inference
        show_progress: Show progress bar
    """
    
    all_preds = [] 
    all_accessions = [] 
    all_alphas = [] # list to store the alpha value from the dynamic gating 
    model.eval()
    
    iterator = tqdm(dataloader, desc="Predicting", disable=not show_progress)
    
    # Pre-allocate list capacity if possible
    try:
        total_samples = len(dataloader.dataset)
        all_preds = [None] * total_samples
        all_accessions = [None] * total_samples
        use_preallocated = True
        idx = 0
    except:
        use_preallocated = False
    
    with torch.inference_mode():
        for batch in iterator: 
            preds = predict_batch(model, batch, device, use_fp16=use_fp16)
            accessions = batch['accession']

            # catch the dynamic weights safely
            if hasattr(model, '_last_alpha_batch') and model._last_alpha_batch is not None:
                batch_alphas = model._last_alpha_batch
                if isinstance(batch_alphas, dict):
                    # tri_stream_dynamic_gate: keys are 'glm', 'arch', 'marker'
                    if not isinstance(all_alphas, dict):
                        all_alphas = {'glm': [], 'arch': [], 'marker': []}
                    for key in ('glm', 'arch', 'marker'):
                        vals = batch_alphas[key]
                        all_alphas[key].extend(vals.tolist() if hasattr(vals, 'tolist') else list(vals))
                elif isinstance(batch_alphas, (int, float)):
                    all_alphas.append(float(batch_alphas))
                elif hasattr(batch_alphas, 'tolist'):
                    alphas_list = batch_alphas.tolist()
                    if isinstance(alphas_list, list):
                        all_alphas.extend(alphas_list)
                    else:
                        all_alphas.append(float(alphas_list))
                elif isinstance(batch_alphas, (list, tuple)):
                    all_alphas.extend([float(alpha) for alpha in batch_alphas])
            
            if use_preallocated:
                batch_size = len(preds)
                all_preds[idx:idx+batch_size] = preds
                all_accessions[idx:idx+batch_size] = accessions
                idx += batch_size
            else:
                all_preds.extend(preds)
                all_accessions.extend(accessions)
    
    # Trim preallocated lists if needed
    if use_preallocated:
        all_preds = [p for p in all_preds if p is not None]
        all_accessions = [a for a in all_accessions if a is not None]
    
    return all_preds, all_accessions, all_alphas



