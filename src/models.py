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
from sequences import embed_window
import torch.nn.functional as F
from transformers import AutoTokenizer, BertConfig, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel
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
            log_pt = F.log_softmax(logits, dim=-1)
            log_pt = log_pt.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            pt = torch.exp(log_pt)
            focal_weight = ((1 - pt) ** self.gamma)
            ce_loss = -log_pt
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
        log_pt = F.log_softmax(logits, dim=-1)
        log_pt = log_pt.gather(1, labels.unsqueeze(-1)).squeeze(-1)
        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = -log_pt
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
    """Factory that returns a collate function which tokenizes + pads sequences."""
    def collate_fn(batch):
        sequences = []
        labels = []
        accessions = []
        features_list = []
        raw_features_list = []
        marker_list = []
        has_features = 'features' in batch[0] and batch[0]['features'] is not None
        has_raw_features = 'raw_features' in batch[0] and batch[0]['raw_features'] is not None
        has_markers = 'marker_indices' in batch[0] and batch[0]['marker_indices'] is not None

        for item in batch:
            sequences.append(item['sequence'])
            labels.append(item['labels'])
            accessions.append(item['accession'])
            if has_features:
                features_list.append(item['features'])
            if has_raw_features:
                raw_features_list.append(item['raw_features'])
            if has_markers:
                marker_list.append(item['marker_indices'])

        labels_tensor = torch.stack(labels)
        features_tensor = torch.stack(features_list) if has_features else None
        raw_features_tensor = torch.stack(raw_features_list) if has_raw_features else None
        marker_tensor = (
            torch.nn.utils.rnn.pad_sequence(marker_list, batch_first=True, padding_value=0)
            if has_markers else None
        )

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
        elif model_type == "genomeocean":
            tokens_out = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                padding_side="right",
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
        if raw_features_tensor is not None:
            result['raw_features'] = raw_features_tensor
        if marker_tensor is not None:
            result['marker_indices'] = marker_tensor
        return result

    return collate_fn


def collate_fn_sequences(batch):
    """Legacy collate that passes raw sequences (tokenization happens in model.forward)."""
    sequences = []
    labels = []
    accessions = []
    features_list = []
    raw_features_list = []
    marker_list = []
    has_features = 'features' in batch[0] and batch[0]['features'] is not None
    has_raw_features = 'raw_features' in batch[0] and batch[0]['raw_features'] is not None
    has_markers = 'marker_indices' in batch[0] and batch[0]['marker_indices'] is not None

    for item in batch:
        sequences.append(item['sequence'])
        labels.append(item['labels'])
        accessions.append(item['accession'])
        if has_features:
            features_list.append(item['features'])
        if has_raw_features:
            raw_features_list.append(item['raw_features'])
        if has_markers:
            marker_list.append(item['marker_indices'])

    labels_tensor = torch.stack(labels)
    features_tensor = torch.stack(features_list) if has_features else None
    raw_features_tensor = torch.stack(raw_features_list) if has_raw_features else None
    marker_tensor = (
        torch.nn.utils.rnn.pad_sequence(marker_list, batch_first=True, padding_value=0)
        if has_markers else None
    )

    result = {
        'sequences': sequences,
        'labels': labels_tensor,
        'features': features_tensor,
        'raw_features': raw_features_tensor,
        'accession': accessions,
    }
    if marker_tensor is not None:
        result['marker_indices'] = marker_tensor
    return result


class GenomeClassifier(nn.Module):
    """Stacking classifier: XGBoost class probabilities gated against a GLM stream.

    The feature branch receives XGBoost out-of-fold probabilities [B, num_classes]
    pre-computed during data loading. A learned sigmoid gate blends those
    probabilities with the GLM logit stream on a per-sample basis.
    """

    def __init__(
        self,
        base_model,
        num_classes: int,
        hidden_size: int,
        tokenizer=None,
        pooling: str = "cls",
        model_type: str = "bibert",
        dropout: float = 0.1,
        use_glm: bool = True,
        use_features: bool = True,
        pooling_heads: int = 4,
        pooling_projection: str = "bottleneck",
        pooling_dropout: float = 0.1,
        debug: bool = False,
        max_length: int = 11904,
        gate_hidden_dim: int = 64,
        glm_drop_rate: float = 0.15,
        feat_drop_rate: float = 0.05,
        use_coarse_head: bool = False,
        prokaryote_idx: int = 2,
        xgb_feature_dim: int = 0,
        gate_marker_dim: int = 0,  # extra gate-only scalars appended after xgb_prob columns
        raw_feature_dim: int = 0,  # dim of raw arch/marker features used as gate input
        **kwargs,  # absorbs legacy args (feature_dim, marker_vocab_size, etc.)
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
        self.feature_dim = xgb_feature_dim if xgb_feature_dim > 0 else num_classes
        self.gate_marker_dim = gate_marker_dim
        self.max_length = max_length
        self.gate_hidden_dim = gate_hidden_dim
        self.glm_drop_rate = glm_drop_rate
        self.feat_drop_rate = feat_drop_rate
        self.prokaryote_idx = prokaryote_idx
        self.raw_feature_dim = raw_feature_dim
        self.feature_integration_mode = "stacking"
        self.loss_log_vars = None
        self.stream_weight_logit = None
        self.classifier = None

        if self.pooling == "learnable":
            raise NotImplementedError("Learnable pooling is not currently implemented in this model version.")

        if self.use_glm:
            self.glm_norm = nn.LayerNorm(hidden_size)
            self.glm_classifier = nn.Linear(hidden_size, num_classes)
        else:
            self.glm_norm = None
            self.glm_classifier = None

        if self.use_glm and self.use_features:
            if raw_feature_dim == 0:
                raise ValueError(
                    "raw_feature_dim must be > 0. "
                    "Provide raw biological features (arch/marker) as gate input."
                )
            _gate_in_dim = raw_feature_dim + gate_marker_dim + num_classes
            self.dynamic_gate = nn.Sequential(
                nn.Linear(_gate_in_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, 1),
            )
        else:
            self.dynamic_gate = None

        # Coarse head: binary prokaryote/eukaryote classifier sharing pooled_normed.
        # Trained with an auxiliary loss; used at inference for confidence flagging.
        if self.use_glm and use_coarse_head:
            self.coarse_classifier = nn.Linear(hidden_size, 2)
        else:
            self.coarse_classifier = None

    def get_stream_weight(self) -> Optional[float]:
        if hasattr(self, '_last_alpha_mean'):
            return self._last_alpha_mean
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sequences=None,
        features=None,
        raw_features=None,  # [B, raw_feature_dim] raw arch/marker features for gate
        marker_indices=None,  # kept for interface compat, unused
        labels=None,
        return_auxiliary_logits=False,
    ):
        """Forward pass.

        Args:
            features: XGBoost class probabilities [B, num_classes] (mixture step only).
            raw_features: Raw biological features [B, raw_feature_dim] used as gate
                input. When provided the gate is decoupled from XGBoost predictions.
                When None and raw_feature_dim==0 the legacy xgb_probs gate is used.
        """
        device = next(self.base_model.parameters()).device

        # Legacy fallback: tokenize inline if raw sequences passed
        if input_ids is None and sequences is not None:
            if self.use_glm:
                if self.model_type == "nucleotidetransformer-v3":
                    tokens_out = self.tokenizer(
                        sequences, return_tensors="pt", truncation=True,
                        max_length=self.max_length, padding=True,
                        pad_to_multiple_of=128, add_special_tokens=False,
                    )
                elif self.model_type == "genomeocean":
                    tokens_out = self.tokenizer(
                        sequences, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_length, padding_side="right",
                    )
                else:
                    tokens_out = self.tokenizer(
                        sequences, return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_length,
                    )
                input_ids = tokens_out["input_ids"]
                attention_mask = tokens_out.get("attention_mask")

        if input_ids is None and sequences is None and self.use_glm:
            raise ValueError("Either input_ids or sequences must be provided when use_glm=True")

        # XGBoost class probabilities [B, feature_dim] from predict_proba, optionally followed
        # by gate_marker_dim scalar columns (e.g. total_marker_freq) that inform the gate
        # but are not used in the probability-space mixture.
        xgb_probs = None
        gate_marker = None
        if features is not None:
            _all_features = features.to(device)
            if self.gate_marker_dim > 0 and _all_features.shape[1] > self.feature_dim:
                xgb_probs = _all_features[:, :self.feature_dim]
                gate_marker = _all_features[:, self.feature_dim:]
                if torch.isnan(gate_marker).any():
                    gate_marker = torch.nan_to_num(gate_marker, nan=0.0)
            else:
                xgb_probs = _all_features


            if torch.isnan(xgb_probs).any():
                raise ValueError(
                    "CRITICAL: NaN values detected in the XGBoost probability tensor! "
                    "This means your upstream XGBoost predictor failed to output probabilities."
                )

        # GLM stream: embed and pool
        if self.use_glm and input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = (
                attention_mask.to(device) if attention_mask is not None
                else torch.ones_like(input_ids)
            )
            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if self.model_type == "genomeocean":
                forward_kwargs["use_cache"] = False
            torch_outs = self.base_model(**forward_kwargs)

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
            if xgb_probs is not None:
                batch_size = xgb_probs.shape[0]
            elif input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = len(sequences)
            pooled = torch.zeros((batch_size, self.hidden_size), device=device)

        # Per-stream dropout during training
        if self.training:
            if torch.rand(1).item() < self.glm_drop_rate:
                pooled = torch.zeros_like(pooled)
            elif xgb_probs is not None and torch.rand(1).item() < self.feat_drop_rate:
                # Simulate XGB uncertainty by sampling from the uniform distribution
                # on the probability simplex (Dirichlet(1,...,1) via the
                # Exp(1)/normalise trick).  This avoids the perfectly-detectable
                # [1/C,...,1/C] constant that the gate MLP could use as a shortcut
                # to ignore XGB without learning genuine uncertainty.  The result
                # still sums to 1.0, keeping NLLLoss stable.
                _exp = -torch.log(torch.rand_like(xgb_probs).clamp(min=1e-10))
                xgb_probs = _exp / _exp.sum(dim=-1, keepdim=True)
                if gate_marker is not None:
                    gate_marker = torch.zeros_like(gate_marker)

        if self.use_glm and self.glm_norm is not None:
            pooled_normed = self.glm_norm(pooled)
            pooled_normed = self.dropout(pooled_normed)
            glm_logits = self.glm_classifier(pooled_normed)
        else:
            pooled_normed = pooled
            glm_logits = None

        # Coarse binary head (prokaryote=0 / eukaryote=1) off the shared representation.
        # Only computed when return_auxiliary_logits=True — avoids a dead forward pass
        # during inference and when use_auxiliary_loss=False.
        coarse_logits = (
            self.coarse_classifier(pooled_normed)
            if (self.coarse_classifier is not None and return_auxiliary_logits)
            else None
        )

        # Binary XGB (e.g. arch_only_xgb1) outputs 2 columns [P(prok), P(euk)].
        # Expand to num_classes so that mixing, the XGB-only fallback, and the
        # auxiliary stream metric all operate in the same probability space.
        # The gate input keeps the original feature_dim (e.g. 2) to match the MLP.
        if xgb_probs is not None and xgb_probs.shape[1] != self.num_classes:
            if xgb_probs.shape[1] == 2:
                n_euk = self.num_classes - 1
                xgb_probs_full = xgb_probs[:, 1:2].expand(-1, self.num_classes).clone() / n_euk
                xgb_probs_full[:, self.prokaryote_idx] = xgb_probs[:, 0]
            else:
                raise ValueError(f"CRITICAL: xgb_probs has {xgb_probs.shape[1]} columns, but num_classes is {self.num_classes}. Expected exactly 2 or {self.num_classes}.")
        else:
            xgb_probs_full = xgb_probs  # already num_classes or None

        if glm_logits is not None and xgb_probs is not None and self.dynamic_gate is not None:
            if raw_features is None:
                raise ValueError(
                    "raw_features must be provided when dynamic_gate is active. "
                    "Pass raw biological features (arch/marker) as gate input."
                )
            rf = raw_features.to(device)
            if rf.dtype != pooled_normed.dtype:
                rf = rf.to(pooled_normed.dtype)
            rf_safe = torch.nan_to_num(rf, nan=0.0)
            rf_scaled = torch.log1p(torch.relu(rf_safe))
            glm_probs_detached = F.softmax(glm_logits.detach(), dim=-1).to(rf_scaled.dtype)
            gate_parts = [rf_scaled, glm_probs_detached]
            if gate_marker is not None:
                gm_safe = torch.nan_to_num(gate_marker, nan=0.0)
                gate_parts.append(torch.log1p(torch.relu(gm_safe)))
            gate_input = torch.cat(gate_parts, dim=-1)
            alpha = torch.sigmoid(self.dynamic_gate(gate_input))
            
            if not self.training:
                self._last_alpha_batch = alpha.detach().cpu().numpy().squeeze()
                self._last_alpha_mean = alpha.mean().item()

            # Proper mixture in probability space — eliminates the logits-vs-probabilities
            # scale mismatch. Output is log-probabilities for use with NLLLoss.
            glm_probs = F.softmax(glm_logits, dim=-1)
            mixed_probs = alpha * glm_probs + (1 - alpha) * xgb_probs_full
            final_logits = torch.log(mixed_probs.clamp(min=1e-7))  # log-probs [B, C]
        elif glm_logits is not None:
            # No XGBoost features available — return log-softmax so the output is
            # always log-probabilities, keeping the downstream loss and probability
            # extraction consistent regardless of whether features were provided.
            final_logits = F.log_softmax(glm_logits, dim=-1)
        else:
            # XGBoost-only fallback: return log-probs for consistency
            final_logits = torch.log(xgb_probs_full.clamp(min=1e-7)) if xgb_probs_full is not None else None

        xgb_log_probs = torch.log(xgb_probs_full.clamp(min=1e-7)) if xgb_probs_full is not None else None
        if return_auxiliary_logits:
            # 5-tuple: (combined_log_probs, glm_raw_logits, xgb_log_probs,
            #           coarse_raw_logits_or_None, loss_log_vars)
            return final_logits, glm_logits, xgb_log_probs, coarse_logits, self.loss_log_vars
        return final_logits

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
            output_attentions=True,
        )
        return attentions, None


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
            pathlib.Path(model_path) / "best_macro_auprc_model" / "config.json",
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
        elif model_type == "genomeocean":
            base_model_name = "DOEJGI/GenomeOcean-4B"
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

        elif model_type == "genomeocean":
            tokenizer = AutoTokenizer.from_pretrained(
                base_tokenizer_name, trust_remote_code=True, padding_side="right"
            )
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                print("flash-attn found — using flash_attention_2")
            except ImportError:
                attn_impl = "eager"
                print("flash-attn not installed — falling back to eager attention")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
            if hasattr(model.config, "hidden_size"):
                hidden_size = model.config.hidden_size
            elif hasattr(model.config, "d_model"):
                hidden_size = model.config.d_model
            else:
                hidden_size = model.get_input_embeddings().weight.shape[1]

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    except Exception as e:
        raise RuntimeError(f"Failed to load base model '{base_model_name}': {e}")

    model.to(device)
    model.eval()

    if model_path:
        best_macro_path = pathlib.Path(model_path) / "best_macro_auprc_model"
        best_macro_legacy_path = pathlib.Path(model_path) / "best_macro_f1_model"
        best_model_path = pathlib.Path(model_path) / "best_model"
        possible_checkpoints = [
            pathlib.Path(model_path) / "training_state.pt",
            best_macro_path / "training_state.pt",
            best_macro_legacy_path / "training_state.pt",
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
