#!/usr/bin/env python3 
"""
modules to handle sequence embedding and manipulation
"""
# imports
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional
from Bio import SeqIO
import math

def generate_chunks(sequences, chunk_size: int = 10000, min_length: int = 1000, strategy: str='uniform-random') -> dict: 
    """
    generate chunks for a dictionary of sequences 
    """ 
    chunks_dict = {}
    for s in list(sequences.keys()):
        if strategy == 'tile': 
            this_chunks = chunk_sequence(sequences[s], chunk_size=chunk_size, min_length=min_length)
        elif strategy == 'uniform-random':  
            this_chunks = uniform_random_chunk_sequence(sequences[s], max_chunk_size=chunk_size, min_chunk_size=min_length) 

        # add chunks to the dictionary 
        for t, u in enumerate(this_chunks):
            chunk_id = s + f"_chunk{t+1}"
            chunks_dict[chunk_id] = u
    return chunks_dict

def uniform_random_chunk_sequence(
    seq:str, 
    max_chunk_size: int = 10000,
    min_chunk_size: int=1000, 
    target_coverage: float = 1.0,
    max_retries: int = 50,
) -> List[str]:
    """
    Use a uniform random sampling strategy to fragment sequences,
    enforcing global 1x coverage math to prevent oversampling biases.
    """
    seq_len = len(seq) 
    chunks = []

    # Strictly ignore anything under the hard minimum
    if seq_len < min_chunk_size:
        return chunks

    # FIX 1: Calculate Global Expected Chunk Size (mu)
    mu_global = (min_chunk_size + max_chunk_size) / 2.0
    
    # FIX 2: Probabilistic Rounding for N
    expected_chunks = (seq_len * target_coverage) / mu_global
    base_chunks = math.floor(expected_chunks)
    prob = expected_chunks - base_chunks
    
    # N becomes base_chunks + 1 (with probability 'prob') or stays base_chunks
    N = base_chunks + (1 if random.random() < prob else 0)
    
    if N == 0:
        return chunks

    # FIX 3: The "Pass-Through" Rule for Tiny Partitions
    pass_through_threshold = min_chunk_size + 500
    if seq_len <= pass_through_threshold:
        if 'N' not in seq.upper():
            # If selected for this epoch, pass the whole partition through N times
            for _ in range(N):
                chunks.append(seq)
        return chunks

    # Normal Chunking for larger genomes
    effective_max = min(max_chunk_size, seq_len)
    
    for _ in range(N):
        for _retry in range(max_retries):
            start = random.randint(0, seq_len - min_chunk_size)
            space_remaining = seq_len - start
            this_max = min(effective_max, space_remaining)
            chunk_length = random.randint(min_chunk_size, this_max)
            chunk = seq[start:start + chunk_length]
            
            # Skip if sequence contains undefined nucleotides
            if 'N' in chunk.upper():
                continue
                
            chunks.append(chunk)
            break
            
    return chunks

def chunk_sequence(
    seq:str, 
    chunk_size: int = 10000, 
    min_length: int = 1000
) -> List[str]:
    """
    Fragment a sequence into non-overlapping chunks of specified size. 
    """ 
    seq_len = len(seq) 
    chunks = [] 

    if seq_len <= chunk_size:
        chunks.append(seq)
        return chunks
    else: 
        start = random.randint(0, chunk_size)
        if start >= min_length:
            chunk1 = seq[0:start]
            chunks.append(chunk1)
        for i in range(start, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            chunk = seq[i:end]
            if len(chunk) >= min_length:
                chunks.append(chunk)    
        return chunks

def load_fasta_sequences(fasta_path: str):
    sequences = []
    accessions = []
    for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
        sequences.append(str(record.seq).upper())
        accessions.append(record.id.split()[0])
    return sequences, accessions

def save_fasta_sequences(sequences: dict, output_path: str):
    with open(output_path, 'w') as f:
        for accession, seq in sequences.items():
            f.write(f">{accession}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

def shuffle_sequence(seq: str) -> str:
    seq_list = list(seq)
    random.shuffle(seq_list)
    return ''.join(seq_list)

def embed_window(
    seq: str,
    tokenizer,
    model,
    device: str,
    pooling: str = "max",
    model_type: str = "bibert",
    return_tensor: bool = False,
    inference_mode: bool = True,
    output_attentions: bool = False,  # <--- NEW ARGUMENT
    debug: bool = False,
):
    """Return a 1-d embedding for the sequence using the specified pooling strategy."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if debug:
        print(f"\n[DEBUG embed_window]")
        print(f"  Input sequence length: {len(seq)} bp")
        print(f"  Model type: {model_type}")

    context = torch.no_grad() if inference_mode else torch.enable_grad()
    
    with context:
        if model_type == "modernbert":
            inputs = tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
            # ModernBERT might not support output_attentions=True in the same way, verify config
            outputs = model(inputs, output_attentions=output_attentions)
            hidden_states = outputs[0]
            attentions = outputs.attentions if output_attentions else None
            
            if pooling == "cls":
                embedding = hidden_states[0, 0, :]
            elif pooling == "mean":
                embedding = hidden_states[0].mean(dim=0)
            elif pooling == "max":
                embedding = torch.max(hidden_states[0], dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

        # --- EXPLICIT NTV3 BLOCK (The Elegant Tokenizer Solution) ---
        elif model_type == "nucleotidetransformer-v3":
            
            # Use the tokenizer's native U-Net padding capabilities
            tokens_out = tokenizer(
                seq, 
                return_tensors="pt", 
                truncation=True,        
                max_length=10112, # needs to be a multiple of 128 and we have max size of 10,000 bp
                padding=True,
                pad_to_multiple_of=128,
                add_special_tokens=False # Matches your successful notebook test
            )
            
            input_ids = tokens_out["input_ids"].to(device)
            if "attention_mask" in tokens_out:
                attention_mask = tokens_out["attention_mask"].to(device)
            else:
                attention_mask = torch.ones_like(input_ids)
            
            torch_outs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )
            
            if hasattr(torch_outs, 'hidden_states') and torch_outs.hidden_states is not None:
                hidden_states = torch_outs.hidden_states[-1]
            else:
                hidden_states = torch_outs[0]

            attentions = torch_outs.attentions if getattr(torch_outs, 'attentions', None) else None

            # Masked Pooling (Automatically ignores the padding added by the tokenizer)
            if pooling == "cls":
                embedding = hidden_states[0, 0, :]
            elif pooling == "mean":
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_emb = torch.sum(hidden_states[0] * mask_expanded[0], dim=0)
                sum_mask = torch.clamp(mask_expanded[0].sum(dim=0), min=1e-9)
                embedding = sum_emb / sum_mask
            elif pooling == "max":
                hs = hidden_states[0].clone()
                hs[attention_mask[0] == 0] = -1e9
                embedding = torch.max(hs, dim=0)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")
    
    # Return logic
    if output_attentions:
        emb_val = embedding if return_tensor else embedding.detach().cpu().numpy()
        return emb_val, attentions
    
    if return_tensor:
        return embedding
    else:
        return embedding.detach().cpu().numpy()


class GenomeDataset(Dataset):
    """Dataset for genome sequences with classification labels."""
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        accessions: List[str],
        tokenizer,
        max_length: int = 100000,
        model_type: str = "bibert",
        features: List[List[float]] = None,
        token_pooling: str = "mean",
        raw_features: List[List[float]] = None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.accessions = accessions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        self.features = features
        self.raw_features = raw_features
        self.token_pooling = token_pooling
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        accession = self.accessions[idx]
        
        if self.features is not None:
            try:
                dense_features = self.features[idx]
            except Exception as e:
                raise ValueError(f"Features are provided but could not be indexed at idx={idx}: {e}")
            dense_tensor = torch.tensor(dense_features, dtype=torch.float)
        else:
            dense_tensor = None

        result = {
            'sequence': seq,
            'labels': torch.tensor(label, dtype=torch.long),
            'accession': accession,
        }

        if dense_tensor is not None:
            result['features'] = dense_tensor

        if self.raw_features is not None:
            try:
                raw_dense = self.raw_features[idx]
            except Exception as e:
                raise ValueError(
                    f"raw_features provided but could not be indexed at idx={idx}: {e}"
                )
            result['raw_features'] = torch.tensor(raw_dense, dtype=torch.float)

        return result