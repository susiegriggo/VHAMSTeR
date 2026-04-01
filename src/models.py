#!/usr/bin/env python3 
"""
modules for glm fine tuning etc.
"""
# imports 
import torch.nn as nn
import torch
import numpy as np
import math
from typing import List, Optional
try:
    from .sequences import embed_window
except ImportError:
    from sequences import embed_window
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig, AutoModelForMaskedLM, AutoModel
import pathlib
import json


class ChunkPoolingLayer(nn.Module):
    """
    Pools multiple chunk embeddings into a single sequence representation.
    Supports mean, max, and learnable attention-based pooling.
    """
    def __init__(self, hidden_dim, pooling_type='mean', num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        
        if pooling_type == 'learnable':
            # Learnable attention weights for chunks
            self.queries = nn.Parameter(torch.empty(num_heads, hidden_dim))
            nn.init.normal_(self.queries, mean=0, std=1/math.sqrt(hidden_dim))
            self.temperatures = nn.Parameter(torch.ones(num_heads))
            
            bottleneck_dim = hidden_dim // 2
            self.projection_block = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, hidden_dim)
            )
            self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, chunk_embeddings, chunk_mask=None):
        if chunk_mask is None:
            chunk_mask = torch.ones(chunk_embeddings.shape[0], chunk_embeddings.shape[1], 
                                   dtype=torch.bool, device=chunk_embeddings.device)
        
        if self.pooling_type == 'mean':
            mask_expanded = chunk_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(chunk_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_type == 'max':
            chunk_embeddings_masked = chunk_embeddings.clone()
            chunk_embeddings_masked[~chunk_mask] = -1e9
            return torch.max(chunk_embeddings_masked, dim=1)[0]
        
        elif self.pooling_type == 'learnable':
            attn_scores = torch.matmul(chunk_embeddings, self.queries.t())
            attn_scores = attn_scores.transpose(1, 2)
            
            scaled_temps = torch.clamp(self.temperatures, min=0.1, max=10.0).view(1, -1, 1)
            attn_scores = attn_scores / scaled_temps
            
            mask_value = -1e4 if chunk_embeddings.dtype == torch.bfloat16 else -1e9
            attn_scores = attn_scores.masked_fill(~chunk_mask.unsqueeze(1), mask_value)
            
            attn_weights = F.softmax(attn_scores, dim=2)
            
            if self.training:
                attn_weights = self.dropout(attn_weights)
            
            pooled_heads = torch.bmm(attn_weights, chunk_embeddings)
            
            concatenated_heads = pooled_heads.reshape(pooled_heads.size(0), -1)
            final_pooled = self.projection_block(concatenated_heads)
            
            return final_pooled
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

class ClassBalancedLoss(nn.Module):
    """
    Implement Class-Balanced Loss as described in https://arxiv.org/abs/1901.05555  
    """
    def __init__(self, beta, samples_per_class, loss_type='softmax', gamma=2.0): 
        super().__init__() 
        self.beta = beta 
        self.sample_per_class = samples_per_class
        self.loss_type = loss_type
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_class)
        effective_num = np.where(effective_num == 0, 1e-8, effective_num)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.class_weights = torch.tensor(weights, dtype=torch.float32) 

    def forward(self, logits, labels): 
        weights = self.class_weights.to(logits.device)
        if self.loss_type == 'softmax': 
            ce_loss = F.cross_entropy(logits, labels,  weight=weights) 
            loss = ce_loss
        elif self.loss_type == 'focal': 
            # Get raw log probabilities (unweighted) so pt is the true probability
            log_pt = F.log_softmax(logits, dim=-1)
            log_pt = log_pt.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            pt = torch.exp(log_pt)
            focal_weight = ((1 - pt) ** self.gamma)
            ce_loss = -log_pt
            # Apply class-balanced weights
            class_weights = weights.gather(0, labels)
            focal_loss = class_weights * focal_weight * ce_loss
            loss = focal_loss.mean()
        else: 
            raise ValueError("Unknown loss type: {}".format(self.loss_type))
        return loss
    
class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        # 1. Get raw log probabilities (unweighted)
        log_pt = F.log_softmax(logits, dim=-1)

        # 2. Gather the log probabilities of the true classes
        log_pt = log_pt.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        pt = torch.exp(log_pt)

        # 3. Calculate focal weight from the TRUE probability
        focal_weight = (1 - pt) ** self.gamma

        # 4. Standard unweighted cross entropy
        ce_loss = -log_pt

        # 5. Apply class weights if they exist
        if self.alpha is not None:
            class_weights = self.alpha.gather(0, labels)
            focal_loss = class_weights * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class MultiHeadLearnablePooling(nn.Module):
    """
    Multi-head learnable pooling with a configurable projection layer.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1, projection_type='bottleneck'):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.projection_type = projection_type
        
        self.queries = nn.Parameter(torch.empty(num_heads, hidden_dim))
        nn.init.normal_(self.queries, mean=0, std=1/math.sqrt(hidden_dim))
        
        self.temperatures = nn.Parameter(torch.ones(num_heads))
        
        if self.projection_type == 'bottleneck':
            bottleneck_dim = hidden_dim // 2
            self.projection_block = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, hidden_dim)
            )
        elif self.projection_type == 'linear':
            self.projection_block = nn.Linear(hidden_dim * num_heads, hidden_dim)
        elif self.projection_type == 'weighted_sum':
            self.projection_block = nn.Parameter(torch.ones(num_heads) / num_heads)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, hidden_states, attention_mask):
        attn_scores = torch.matmul(hidden_states, self.queries.t())
        attn_scores = attn_scores.transpose(1, 2)
        
        scaled_temps = torch.clamp(self.temperatures, min=0.1, max=10.0).view(1, -1, 1)
        attn_scores = attn_scores / scaled_temps
        
        mask_value = -1e4 if hidden_states.dtype == torch.bfloat16 else -1e9
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, mask_value)
        
        attn_weights = F.softmax(attn_scores, dim=2)
        
        if self.training:
            attn_weights = self.dropout(attn_weights)
        
        pooled_heads = torch.bmm(attn_weights, hidden_states)
        
        if self.projection_type == 'weighted_sum':
            final_pooled = torch.sum(
                pooled_heads * self.projection_block.view(1, -1, 1),
                dim=1
            )
        else: 
            concatenated_heads = pooled_heads.view(pooled_heads.size(0), -1)
            final_pooled = self.projection_block(concatenated_heads)
        
        return final_pooled

def make_collate_fn(tokenizer, model_type, max_length):
    """Factory that returns a collate function which tokenizes + pads sequences.
    
    Tokenization happens here (once per batch) instead of inside model.forward(),
    keeping the forward pass a pure tensor-in/tensor-out function.
    """
    def collate_fn(batch):
        sequences = []
        labels = []
        accessions = []
        features_list = []
        has_features = 'features' in batch[0] and batch[0]['features'] is not None

        for item in batch:
            sequences.append(item['sequence'])
            labels.append(item['labels'])
            accessions.append(item['accession'])
            if has_features:
                features_list.append(item['features'])

        labels_tensor = torch.stack(labels)
        features_tensor = torch.stack(features_list) if has_features else None

        # Tokenize the batch of sequences
        if model_type == "nucleotidetransformer-v3":
            tokens_out = tokenizer(
                sequences,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                pad_to_multiple_of=128,
                add_special_tokens=False,
            )
        else:
            tokens_out = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        result = {
            'input_ids': tokens_out['input_ids'],
            'attention_mask': tokens_out.get('attention_mask', torch.ones_like(tokens_out['input_ids'])),
            'labels': labels_tensor,
            'accession': accessions,
        }
        if features_tensor is not None:
            result['features'] = features_tensor
        return result

    return collate_fn


# Backward-compatible alias for code that doesn't need tokenization in collate
def collate_fn_sequences(batch):
    """Legacy collate that passes raw sequences (tokenization happens in model.forward).
    Prefer make_collate_fn() for new code.
    """
    sequences = []
    labels = []
    accessions = []
    features_list = []
    has_features = 'features' in batch[0] and batch[0]['features'] is not None

    for item in batch:
        sequences.append(item['sequence'])
        labels.append(item['labels'])
        accessions.append(item['accession'])
        if has_features:
            features_list.append(item['features'])

    labels_tensor = torch.stack(labels)
    features_tensor = torch.stack(features_list) if has_features else None

    return {
        'sequences': sequences,
        'labels': labels_tensor,
        'features': features_tensor,
        'accession': accessions
    }

class GenomeClassifier(nn.Module):
    """Wrapper that adds a classification head to genomic language models."""

    def __init__(
        self,
        base_model,
        num_classes: int,
        hidden_size: int,
        tokenizer=None,
        pooling: str = "cls",
        model_type: str = "bibert",
        dropout: float = 0.1,
        feature_dim: int = 0,
        use_features: bool = True,
        use_glm: bool = True,
        pooling_heads: int = 4,
        pooling_projection: str = "bottleneck",
        pooling_dropout: float = 0.1,
        debug: bool = False,
        feature_integration_mode: str = "concat", 
        stream_weight_init: float = 0.0, 
        use_learnable_aux_loss: bool = False,
        max_length: int = 11904, 
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.model_type = model_type
        self.dropout = nn.Dropout(dropout)
        self.use_features = use_features
        self.use_glm = use_glm
        self.hidden_size = hidden_size
        self.debug = debug
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.feature_integration_mode = feature_integration_mode
        self.use_learnable_aux_loss = use_learnable_aux_loss
        self.max_length = max_length 
        self.gate_hidden_dim = gate_hidden_dim
        
        if self.pooling == "learnable":
            print("Warning: 'learnable' pooling not fully supported with embed_window. Using 'mean' instead.")
            self.pooling = "mean"
        
        if self.feature_integration_mode in ["dual_stream", "dual_stream_dynamic_gate"]:
            if self.use_features and feature_dim > 0:
                self.feature_norm = None  # StandardScaler handles normalization; LayerNorm over 14 dims is too noisy
                self.feature_mlp = nn.Sequential( # MLP for making a prediction based on the pyrodigal features 
                    nn.Linear(feature_dim, 32),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, num_classes)
                )
            else:
                self.feature_norm = None
                self.feature_mlp = None

            if self.use_glm:
                self.glm_norm = nn.LayerNorm(hidden_size)
                self.glm_classifier = nn.Linear(hidden_size, num_classes)
            else:
                self.glm_norm = None
                self.glm_classifier = None
            
            if self.use_glm and self.use_features and feature_dim > 0:
               
                if self.feature_integration_mode == "dual_stream_dynamic_gate":
                    # Learns a custom weight based on the input features + sequence
                    #self.dynamic_gate = nn.Linear(hidden_size + feature_dim, 1)
                    self.dynamic_gate = nn.Sequential(nn.Linear(hidden_size + feature_dim, self.gate_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.gate_hidden_dim, 1)) # use Sequential for a tiny MLP gate instead of just a linear layer - hard coded to 64 dims for now - this is the gating network
                    self.stream_weight_logit = None
                else:
                    self.stream_weight_logit = nn.Parameter(torch.tensor(stream_weight_init))
                    self.dynamic_gate = None
                
                if self.use_learnable_aux_loss:
                    self.loss_log_vars = nn.Parameter(torch.zeros(3))
                else:
                    self.loss_log_vars = None
            else:
                self.stream_weight_logit = None
                self.dynamic_gate = None
                self.loss_log_vars = None
                
            self.classifier = None 

        else: 
            self.feature_mlp = None
            self.glm_classifier = None
            self.stream_weight_logit = None
            self.dynamic_gate = None
            self.loss_log_vars = None
            self.classifier = nn.Linear(hidden_size + (feature_dim if self.use_features else 0), num_classes)

    def get_stream_weight(self) -> Optional[float]:
        if self.stream_weight_logit is not None:
            with torch.no_grad():
                return torch.sigmoid(self.stream_weight_logit).item()
        elif hasattr(self, '_last_alpha_mean'):
            # Return the average alpha from the most recent batch if using dynamic gating
            return self._last_alpha_mean
        return None

    def forward(self, input_ids=None, attention_mask=None, sequences=None, features=None, labels=None, return_auxiliary_logits=False):
        """Forward pass.
        
        Accepts pre-tokenized input_ids + attention_mask (preferred, from make_collate_fn)
        or raw sequences (legacy fallback, tokenized inline).
        """
        device = next(self.base_model.parameters()).device
        
        # Legacy fallback: if raw sequences are passed, tokenize inline
        if input_ids is None and sequences is not None:
            if self.use_glm:
                if self.model_type == "nucleotidetransformer-v3":
                    tokens_out = self.tokenizer(
                        sequences, 
                        return_tensors="pt", 
                        truncation=True,        
                        max_length=self.max_length,
                        padding=True,
                        pad_to_multiple_of=128,
                        add_special_tokens=False
                    )
                else:
                    tokens_out = self.tokenizer(
                        sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length
                    )
                input_ids = tokens_out["input_ids"]
                attention_mask = tokens_out.get("attention_mask")
        
        if input_ids is None and sequences is None and self.use_glm:
            raise ValueError("Either input_ids or sequences must be provided when use_glm=True")
        
        if self.use_glm and input_ids is not None:
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids)
                
            torch_outs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            if hasattr(torch_outs, 'hidden_states') and torch_outs.hidden_states is not None:
                hidden_states = torch_outs.hidden_states[-1]
            else:
                hidden_states = torch_outs[0]

            if self.pooling == "cls":
                pooled = hidden_states[:, 0, :]
            elif self.pooling == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_emb = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_emb / sum_mask
            elif self.pooling == "max":
                hs = hidden_states.clone()
                hs[attention_mask == 0] = -1e9
                pooled = torch.max(hs, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        else:
            # Derive batch size from whatever is available
            if input_ids is not None:
                batch_size = input_ids.shape[0]
            elif features is not None:
                batch_size = features.shape[0]
            elif sequences is not None:
                batch_size = len(sequences)
            else:
                raise ValueError("Cannot determine batch size: no input_ids, features, or sequences provided")
            pooled = torch.zeros((batch_size, self.hidden_size), device=device)
        
        if self.feature_integration_mode in ["dual_stream", "dual_stream_dynamic_gate"]:
            glm_logits = None
            feature_logits = None

            if self.use_features and features is not None and self.feature_mlp is not None:
                if torch.isnan(features).any():
                    features = torch.nan_to_num(features, nan=0.0)
                # Features are pre-scaled using StandardScaler in fine_tune_glm.py
                feature_logits = self.feature_mlp(features) 

            if self.use_glm and self.glm_classifier is not None:
                pooled_normed = self.glm_norm(pooled)
                pooled_normed = self.dropout(pooled_normed)
                glm_logits = self.glm_classifier(pooled_normed)
            
            if glm_logits is not None and feature_logits is not None:
             
                if self.feature_integration_mode == "dual_stream_dynamic_gate":
                    # Input the normalized GLM embeddings and pre-scaled features to the gate network
                    gate_input = torch.cat([pooled_normed, features], dim=-1)
                    dynamic_logits = self.dynamic_gate(gate_input)
                    alpha = torch.sigmoid(dynamic_logits) # Shape: [batch_size, 1]

                    # Only track alpha diagnostics during evaluation to avoid CPU transfers during training
                    if not self.training:
                        self._last_alpha_batch = alpha.detach().cpu().numpy().squeeze()
                        self._last_alpha_mean = alpha.mean().item()
                else:
                    alpha = torch.sigmoid(self.stream_weight_logit)
                
                final_logits = alpha * glm_logits + (1 - alpha) * feature_logits
            elif glm_logits is not None:
                final_logits = glm_logits
            elif feature_logits is not None:
                final_logits = feature_logits
            else:
                final_logits = torch.zeros(len(sequences), self.num_classes, device=device)
            
            if return_auxiliary_logits:
                return final_logits, glm_logits, feature_logits, self.loss_log_vars
            return final_logits

        else:
            if self.use_features and features is not None:
                pooled = torch.cat([pooled, features], dim=-1)
                
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
            
            if return_auxiliary_logits:
                return logits, None, None, None
            return logits

    def extract_interpretability_data(self, sequence, features=None):
        device = next(self.base_model.parameters()).device
        
        embedding, attentions = embed_window(
            seq=sequence,
            tokenizer=self.tokenizer,
            model=self.base_model,
            device=device,
            pooling=self.pooling,
            model_type=self.model_type,
            return_tensor=True,
            inference_mode=True,
            output_attentions=True
        )
        
        feature_weights = None
        if self.feature_integration_mode in ["dual_stream", "dual_stream_dynamic_gate"]:
            if self.use_features and self.feature_mlp is not None:
                # Index 3 is the output Linear layer (feature_dim -> 32 -> GELU -> Dropout -> num_classes)
                feature_weights = self.feature_mlp[3].weight.detach().cpu().numpy()
        else:
            classifier_weights = self.classifier.weight.detach().cpu().numpy()
            if self.use_features and features is not None:
                feature_weights = classifier_weights[:, -self.feature_dim:]

        return attentions, feature_weights

# [load_model_and_tokenizer remains unchanged]
def load_model_and_tokenizer(
    model_path: Optional[str] = None,
    model_type: str = "bibert",
    pooling: str = "mean",
    max_length: int = 128,
    pooling_heads: int = 4,
    pooling_projection: str = "bottleneck",
    pooling_dropout: float = 0.1,
    base_model_name: Optional[str] = None,
    base_tokenizer_name: Optional[str] = None,
) -> tuple:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if model_path:
        possible_configs = [
            pathlib.Path(model_path) / "config.json",
            pathlib.Path(model_path) / "best_macro_f1_model" / "config.json",
            pathlib.Path(model_path) / "best_model" / "config.json"
        ]
        
        for config_path in possible_configs:
            if config_path.exists():
                print(f"Loading configuration from {config_path}")
                try:
                    with open(config_path) as f:
                        train_config = json.load(f)
                    
                    if train_config.get('model_type'):
                        model_type = train_config['model_type']
                        print(f"  Using model_type from config: {model_type}")
                    if train_config.get('model'):
                        base_model_name = train_config['model']
                        print(f"  Using base model from config: {base_model_name}")
                    if train_config.get('tokenizer'):
                        base_tokenizer_name = train_config['tokenizer']
                        print(f"  Using tokenizer from config: {base_tokenizer_name}")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load config from {config_path}: {e}")

    if base_model_name is None:
        if model_type == "bibert":
            base_model_name = "Lancelot53/birnabert-2ep"
        elif model_type == "modernbert":
            base_model_name = "RaphaelMourad/ModernBert-DNA-v1-37M-virus"
        elif model_type == "nucleotidetransformer" or model_type == "esm":
            base_model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        elif model_type == "nucleotidetransformer-v3":
            base_model_name = "InstaDeepAI/NTv3_100M_pre"
        else:
            raise ValueError(f"Cannot set default base model. Unknown model_type: {model_type}")
    
    if base_tokenizer_name is None:
        if model_type == "bibert":
            base_tokenizer_name = "buetnlpbio/birna-tokenizer"
        else:
            base_tokenizer_name = base_model_name

    print(f"Loading base model: {base_model_name}")
    print(f"Loading tokenizer: {base_tokenizer_name}")
    
    model = None 
    
    try:
        if model_type == "bibert":
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
            config = BertConfig.from_pretrained(base_model_name)
            config.alibi_starting_size = max_length
            model = AutoModelForMaskedLM.from_pretrained(
                base_model_name, 
                config=config,
                trust_remote_code=True
            )
            model.cls = nn.Identity()
            hidden_size = config.hidden_size
            
        elif model_type == "modernbert":
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
            model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
            hidden_size = model.config.hidden_size
            
        elif model_type == "nucleotidetransformer" or model_type == "esm":
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(base_model_name, trust_remote_code=True)
            hidden_size = model.config.hidden_size

        elif model_type == "nucleotidetransformer-v3":
            tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name, trust_remote_code=True)
            model = AutoModelForMaskedLM.from_pretrained(base_model_name, trust_remote_code=True)
            
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, "d_model"):
                hidden_size = model.config.d_model
            elif hasattr(model.config, "embed_dim"):
                hidden_size = model.config.embed_dim
            else:
                hidden_size = model.get_input_embeddings().weight.shape[1]
                
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load base model '{base_model_name}': {e}")

    model.to(device)
    model.eval()
    
    if model_path:
        best_macro_path = pathlib.Path(model_path) / "best_macro_f1_model"
        best_model_path = pathlib.Path(model_path) / "best_model"
        possible_checkpoints = [
            pathlib.Path(model_path) / "training_state.pt",
            best_macro_path / "training_state.pt",
            best_model_path / "training_state.pt"
        ]
        
        checkpoint_path = None
        for p in possible_checkpoints:
            if p.exists():
                checkpoint_path = p
                break
        
        if checkpoint_path:
            print(f"Loading fine-tuned weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            base_model_state = {}
            for key, value in state_dict.items():
                if key.startswith('base_model.'):
                    new_key = key[len('base_model.'):]
                    base_model_state[new_key] = value
            
            if base_model_state:
                model.load_state_dict(base_model_state, strict=False)
                print(f"  Loaded {len(base_model_state)} base model parameters")
            else:
                print("  Warning: No base_model weights found in checkpoint")
        else: 
            print(f"No checkpoint found. Using raw pre-trained base model.")

    pooling_layer = None
    if pooling == "learnable":
        pooling_layer = MultiHeadLearnablePooling(
            hidden_size,
            num_heads=pooling_heads,
            dropout_rate=pooling_dropout,
            projection_type=pooling_projection,
        )
        pooling_layer.to(device)
        pooling_layer.eval()
    
    return model, tokenizer, pooling_layer, device, hidden_size