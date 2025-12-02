"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)



        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfAttentionMerged(CausalSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x):
        # print("merged attn")
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, T, n_head, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        # Create two sets of overlapping merged pairs to avoid information leak:
        # Even pairs: (0,1), (2,3), (4,5), ... -> merged at positions 0, 2, 4, ...
        # Odd pairs: (1,2), (3,4), (5,6), ... -> merged at positions 1, 3, 5, ...
        # For token j, use attention output from merge(j-1, j)
        
        # Even pairs: merge (2i, 2i+1) for i=0,1,2,...
        T_even_pairs = (T - 1) // 2 * 2 + 1 if T > 0 else 0  # Last even index that can form a pair
        # Ensure T_even_pairs is even for proper reshaping into pairs
        T_even_pairs = (T_even_pairs // 2) * 2
        if T_even_pairs >= 2:
            num_even_merges = T_even_pairs // 2
            k_even = k[:, :T_even_pairs].view(B, num_even_merges, 2, self.n_head, C // self.n_head).mean(dim=2)  # (B, num_even_merges, nh, hs)
            q_even = q[:, :T_even_pairs].view(B, num_even_merges, 2, self.n_head, C // self.n_head).mean(dim=2)
            v_even = v[:, :T_even_pairs].view(B, num_even_merges, 2, self.n_head, C // self.n_head).mean(dim=2)
        else:
            k_even = torch.empty(B, 0, self.n_head, C // self.n_head, device=k.device, dtype=k.dtype)
            q_even = torch.empty(B, 0, self.n_head, C // self.n_head, device=q.device, dtype=q.dtype)
            v_even = torch.empty(B, 0, self.n_head, C // self.n_head, device=v.device, dtype=v.dtype)
            num_even_merges = 0
        
        # Odd pairs: merge (2i-1, 2i) for i=1,2,3,... (i.e., (1,2), (3,4), (5,6), ...)
        T_odd_start = 1
        T_odd_end = T if T % 2 == 0 else T - 1  # Last odd index that can form a pair
        # Ensure we have an even number of elements for proper reshaping into pairs
        T_odd_end = T_odd_start + ((T_odd_end - T_odd_start) // 2) * 2
        if T_odd_end > T_odd_start:
            num_odd_merges = (T_odd_end - T_odd_start) // 2
            k_odd = k[:, T_odd_start:T_odd_end].view(B, num_odd_merges, 2, self.n_head, C // self.n_head).mean(dim=2)  # (B, num_odd_merges, nh, hs)
            q_odd = q[:, T_odd_start:T_odd_end].view(B, num_odd_merges, 2, self.n_head, C // self.n_head).mean(dim=2)
            v_odd = v[:, T_odd_start:T_odd_end].view(B, num_odd_merges, 2, self.n_head, C // self.n_head).mean(dim=2)
        else:
            k_odd = torch.empty(B, 0, self.n_head, C // self.n_head, device=k.device, dtype=k.dtype)
            q_odd = torch.empty(B, 0, self.n_head, C // self.n_head, device=q.device, dtype=q.dtype)
            v_odd = torch.empty(B, 0, self.n_head, C // self.n_head, device=v.device, dtype=v.dtype)
            num_odd_merges = 0
        
        # Transpose to (B, nh, T_merged, hs) for attention
        T_even_attn = num_even_merges
        T_odd_attn = num_odd_merges
        
        y_even = None
        y_odd = None
        
        # Attention on even pairs
        if T_even_attn > 0:
            k_even_t = k_even.transpose(1, 2)  # (B, nh, T_even_attn, hs)
            q_even_t = q_even.transpose(1, 2)
            v_even_t = v_even.transpose(1, 2)
            
            if self.flash:
                y_even = torch.nn.functional.scaled_dot_product_attention(q_even_t, k_even_t, v_even_t, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att_even = (q_even_t @ k_even_t.transpose(-2, -1)) * (1.0 / math.sqrt(k_even_t.size(-1)))
                att_even = att_even.masked_fill(self.bias[:,:,:T_even_attn,:T_even_attn] == 0, float('-inf'))
                att_even = F.softmax(att_even, dim=-1)
                att_even = self.attn_dropout(att_even)
                y_even = att_even @ v_even_t
            y_even = y_even.transpose(1, 2)  # (B, T_even_attn, nh, hs)
        
        # Attention on odd pairs
        if T_odd_attn > 0:
            k_odd_t = k_odd.transpose(1, 2)  # (B, nh, T_odd_attn, hs)
            q_odd_t = q_odd.transpose(1, 2)
            v_odd_t = v_odd.transpose(1, 2)
            
            if self.flash:
                y_odd = torch.nn.functional.scaled_dot_product_attention(q_odd_t, k_odd_t, v_odd_t, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att_odd = (q_odd_t @ k_odd_t.transpose(-2, -1)) * (1.0 / math.sqrt(k_odd_t.size(-1)))
                att_odd = att_odd.masked_fill(self.bias[:,:,:T_odd_attn,:T_odd_attn] == 0, float('-inf'))
                att_odd = F.softmax(att_odd, dim=-1)
                att_odd = self.attn_dropout(att_odd)
                y_odd = att_odd @ v_odd_t
            y_odd = y_odd.transpose(1, 2)  # (B, T_odd_attn, nh, hs)
        
        # Combine attention outputs: for token j, use attention from merge(j-1, j)
        # Even pairs: merge(2i, 2i+1) -> merge i contains tokens 2i and 2i+1
        # Odd pairs: merge(2i-1, 2i) -> merge i contains tokens 2i-1 and 2i (for i>=1, so tokens 1,2 then 3,4, etc.)
        # For token j:
        #   - If j is even: merge(j-1, j) is in odd-pair set at position (j-2)//2 (since odd pairs start at token 1)
        #   - If j is odd: merge(j-1, j) is in even-pair set at position (j-1)//2
        # Token 0: compute attention separately on token 0 only (no merge, no leakage)
        # Determine output dtype from attention outputs (handles mixed precision autocast)
        if y_even is not None:
            output_dtype = y_even.dtype
        elif y_odd is not None:
            output_dtype = y_odd.dtype
        else:
            output_dtype = q.dtype
        y = torch.zeros(B, T, self.n_head, C // self.n_head, device=x.device, dtype=output_dtype)
        
        # Token 0: compute attention on token 0 alone (no merge to avoid leakage from token 1)
        if T > 0:
            q0 = q[:, 0:1].transpose(1, 2)  # (B, nh, 1, hs)
            k0 = k[:, 0:1].transpose(1, 2)  # (B, nh, 1, hs)
            v0 = v[:, 0:1].transpose(1, 2)  # (B, nh, 1, hs)
            
            if self.flash:
                y0 = torch.nn.functional.scaled_dot_product_attention(q0, k0, v0, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att0 = (q0 @ k0.transpose(-2, -1)) * (1.0 / math.sqrt(k0.size(-1)))  # (B, nh, 1, 1)
                att0 = att0.masked_fill(self.bias[:,:,:1,:1] == 0, float('-inf'))
                att0 = F.softmax(att0, dim=-1)
                att0 = self.attn_dropout(att0)
                y0 = att0 @ v0  # (B, nh, 1, hs)
            y0_reshaped = y0.transpose(1, 2).squeeze(1)  # (B, nh, hs)
            y[:, 0] = y0_reshaped.to(dtype=y.dtype)  # Ensure dtype match
        
        # For tokens j >= 1: use merge(j-1, j) - this ensures no leakage as merge only includes tokens <= j
        # Use vectorized operations instead of loop
        if T > 1:
            # Even tokens: j=2,4,6,... use odd merges
            even_indices = torch.arange(2, T, 2, device=x.device)  # [2, 4, 6, ...]
            if len(even_indices) > 0:
                even_merge_indices = (even_indices - 2) // 2  # [0, 1, 2, ...]
                valid_even_mask = (even_merge_indices >= 0) & (even_merge_indices < T_odd_attn)
                if valid_even_mask.any():
                    valid_even_idx = even_indices[valid_even_mask]
                    valid_even_merge_idx = even_merge_indices[valid_even_mask]
                    y[:, valid_even_idx] = y_odd[:, valid_even_merge_idx].to(dtype=y.dtype)  # Ensure dtype match
                # Handle invalid even indices by copying from previous token
                invalid_even_mask = ~valid_even_mask
                if invalid_even_mask.any():
                    invalid_even_idx = even_indices[invalid_even_mask]
                    y[:, invalid_even_idx] = y[:, invalid_even_idx - 1]
            
            # Odd tokens: j=1,3,5,... use even merges
            odd_indices = torch.arange(1, T, 2, device=x.device)  # [1, 3, 5, ...]
            if len(odd_indices) > 0:
                odd_merge_indices = (odd_indices - 1) // 2  # [0, 1, 2, ...]
                valid_odd_mask = odd_merge_indices < T_even_attn
                if valid_odd_mask.any():
                    valid_odd_idx = odd_indices[valid_odd_mask]
                    valid_odd_merge_idx = odd_merge_indices[valid_odd_mask]
                    y[:, valid_odd_idx] = y_even[:, valid_odd_merge_idx].to(dtype=y.dtype)  # Ensure dtype match
                # Handle invalid odd indices by copying from previous token
                invalid_odd_mask = ~valid_odd_mask
                if invalid_odd_mask.any():
                    invalid_odd_idx = odd_indices[invalid_odd_mask]
                    y[:, invalid_odd_idx] = y[:, invalid_odd_idx - 1]
        
        y = y.contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalSelfAttentionShifted(CausalSelfAttention):

    def __init__(self, config):
        super().__init__(config)
        
    def forward(self, x):
        # print("shifted attn")
        B, T, C = x.size()
        
        # Store original input for residual connection
        x_orig = x
        
        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape to (B, T, n_head, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        # Merge pairs: (0,1), (2,3), (4,5), etc.
        # Number of pairs we can form
        num_pairs = T // 2
        if num_pairs == 0:
            # Not enough tokens to merge pairs, fall back to regular attention
            if T == 1:
                q_t = q[:, 0:1].transpose(1, 2)  # (B, nh, 1, hs)
                k_t = k[:, 0:1].transpose(1, 2)
                v_t = v[:, 0:1].transpose(1, 2)
                
                if self.flash:
                    y = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
                else:
                    att = (q_t @ k_t.transpose(-2, -1)) * (1.0 / math.sqrt(k_t.size(-1)))
                    att = att.masked_fill(self.bias[:,:,:1,:1] == 0, float('-inf'))
                    att = F.softmax(att, dim=-1)
                    att = self.attn_dropout(att)
                    y = att @ v_t
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                y = self.resid_dropout(self.c_proj(y))
                return y + x_orig
            else:
                return x_orig
        
        # Extract pairs: tokens at indices (0,1), (2,3), (4,5), etc.
        pair_start_idx = 0
        pair_end_idx = pair_start_idx + num_pairs * 2
        pairs_tokens = k[:, pair_start_idx:pair_end_idx]  # (B, num_pairs*2, nh, hs)
        
        # Reshape to merge pairs: (B, num_pairs, 2, nh, hs) -> (B, num_pairs, nh, hs)
        k_merged = pairs_tokens.view(B, num_pairs, 2, self.n_head, C // self.n_head).mean(dim=2)
        q_merged = q[:, pair_start_idx:pair_end_idx].view(B, num_pairs, 2, self.n_head, C // self.n_head).mean(dim=2)
        v_merged = v[:, pair_start_idx:pair_end_idx].view(B, num_pairs, 2, self.n_head, C // self.n_head).mean(dim=2)
        
        # Transpose for attention: (B, nh, num_pairs, hs)
        k_merged_t = k_merged.transpose(1, 2)
        q_merged_t = q_merged.transpose(1, 2)
        v_merged_t = v_merged.transpose(1, 2)
        
        # Do attention at 1/2 length (on merged tokens)
        if self.flash:
            y_merged = torch.nn.functional.scaled_dot_product_attention(q_merged_t, k_merged_t, v_merged_t, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att_merged = (q_merged_t @ k_merged_t.transpose(-2, -1)) * (1.0 / math.sqrt(k_merged_t.size(-1)))
            att_merged = att_merged.masked_fill(self.bias[:,:,:num_pairs,:num_pairs] == 0, float('-inf'))
            att_merged = F.softmax(att_merged, dim=-1)
            att_merged = self.attn_dropout(att_merged)
            y_merged = att_merged @ v_merged_t
        
        # Transpose back: (B, num_pairs, nh, hs)
        y_merged = y_merged.transpose(1, 2)
        
        # Shift forward by 1 token position: merged pair (0,1) output -> positions (1,2), merged pair (2,3) -> (3,4), etc.
        # Expand back 1->2 vectors: each merged token becomes 2 tokens
        y_expanded = y_merged.unsqueeze(2).expand(B, num_pairs, 2, self.n_head, C // self.n_head)  # (B, num_pairs, 2, nh, hs)
        y_expanded = y_expanded.contiguous().view(B, num_pairs * 2, self.n_head, C // self.n_head)  # (B, num_pairs*2, nh, hs)
        
        # Create output tensor and place expanded outputs at shifted positions (1,2), (3,4), (5,6), ...
        y = torch.zeros(B, T, self.n_head, C // self.n_head, device=x.device, dtype=y_merged.dtype)
        
        # Handle token 0 separately (no merged pair updates it)
        if T > 0:
            q0 = q[:, 0:1].transpose(1, 2)  # (B, nh, 1, hs)
            k0 = k[:, 0:1].transpose(1, 2)
            v0 = v[:, 0:1].transpose(1, 2)
            
            if self.flash:
                y0 = torch.nn.functional.scaled_dot_product_attention(q0, k0, v0, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att0 = (q0 @ k0.transpose(-2, -1)) * (1.0 / math.sqrt(k0.size(-1)))
                att0 = att0.masked_fill(self.bias[:,:,:1,:1] == 0, float('-inf'))
                att0 = F.softmax(att0, dim=-1)
                att0 = self.attn_dropout(att0)
                y0 = att0 @ v0
            y[:, 0] = y0.transpose(1, 2).squeeze(1)
        
        # Place expanded outputs at shifted positions: merged pair (0,1) -> positions (1,2), merged pair (2,3) -> (3,4), etc.
        output_start_idx = 1  # Start at position 1 (merged pair (0,1) updates positions 1,2)
        output_end_idx = min(output_start_idx + num_pairs * 2, T)
        if output_start_idx < T:
            y[:, output_start_idx:output_end_idx] = y_expanded[:, :(output_end_idx - output_start_idx)]
        
        # Handle any remaining tokens that weren't part of pairs (e.g., if T is odd and last token wasn't paired)
        if output_end_idx < T:
            # Do regular attention on remaining tokens
            q_remaining = q[:, output_end_idx:].transpose(1, 2)
            k_remaining = k[:, output_end_idx:].transpose(1, 2)
            v_remaining = v[:, output_end_idx:].transpose(1, 2)
            
            if self.flash:
                y_remaining = torch.nn.functional.scaled_dot_product_attention(q_remaining, k_remaining, v_remaining, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                T_remaining = T - output_end_idx
                att_remaining = (q_remaining @ k_remaining.transpose(-2, -1)) * (1.0 / math.sqrt(k_remaining.size(-1)))
                att_remaining = att_remaining.masked_fill(self.bias[:,:,:T_remaining,:T_remaining] == 0, float('-inf'))
                att_remaining = F.softmax(att_remaining, dim=-1)
                att_remaining = self.attn_dropout(att_remaining)
                y_remaining = att_remaining @ v_remaining
            y[:, output_end_idx:] = y_remaining.transpose(1, 2)
        
        # Re-assemble all head outputs side by side
        y = y.contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        # Add to residual stream
        return y + x_orig

class CausalSelfAttentionMergedHierarchy(CausalSelfAttention):
    """
    Hierarchical attention with:
      - Multi-level merged K/V (levels 0..kappa), where level i has blocks of size 2^i
        built directly from the base sequence (no recursive dependency).
      - Local window: for a query at position t, the last `local_window` tokens
        are forced to use tier-0 (no merging that includes them).
      - Alpha gating: for tokens outside that local window, coarser levels are only
        allowed when they are far enough away, according to:

            d = window_start(t) - block_end
            level i allowed  <=>  d >= alpha ** i   (for i > 0)

        where window_start(t) = max(0, t + 1 - local_window).
    """

    def __init__(self, config):
        super().__init__(config)
        self.kappa = getattr(config, "kappa", 3)
        self.alpha = float(getattr(config, "alpha", 2.0))
        self.local_window = getattr(config, "local_window", 128)

    @staticmethod
    def _build_hierarchy(k, v, kappa):
        """
        Build hierarchy levels directly from base K/V.

        Inputs:
          k, v : [B, T, H, D]  (level 0, one token per position)

        For level i > 0:
          - block size S = 2^i
          - number of blocks T_i = T // S (only full blocks)
          - j-th block spans [j*S, (j+1)*S - 1] in the original sequence.

        Returns:
          levels_k, levels_v : list of [B, T_i, H, D] per level (0..L-1)
          starts_lvl, ends_lvl: list of [T_i] spans per level.
        """
        device = k.device
        B, T, H, D = k.shape

        base_k, base_v = k, v

        levels_k = [base_k]
        levels_v = [base_v]
        starts_lvl = [torch.arange(T, device=device)]
        ends_lvl   = [torch.arange(T, device=device)]

        for i in range(1, kappa + 1):
            block_size = 1 << i  # 2^i
            n_blocks = T // block_size
            if n_blocks == 0:
                break

            slice_len = n_blocks * block_size
            # [B, slice_len, H, D] -> [B, n_blocks, block_size, H, D] -> mean over block
            k_i = base_k[:, :slice_len].view(B, n_blocks, block_size, H, D).mean(dim=2)
            v_i = base_v[:, :slice_len].view(B, n_blocks, block_size, H, D).mean(dim=2)

            starts_i = torch.arange(n_blocks, device=device) * block_size
            ends_i   = starts_i + (block_size - 1)

            levels_k.append(k_i)
            levels_v.append(v_i)
            starts_lvl.append(starts_i)
            ends_lvl.append(ends_i)

        return levels_k, levels_v, starts_lvl, ends_lvl

    def forward(self, x):
        """
        x : [B, T, C]
        returns : [B, T, C]
        """
        B, T, C = x.size()
        device = x.device
        H = self.n_head
        D = C // H

        if T == 0:
            return x

        # --- Base Q,K,V ---
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # --- Build hierarchy from base K,V ---
        levels_k, levels_v, starts_lvl, ends_lvl = self._build_hierarchy(k, v, self.kappa)
        level_lengths = [lk.size(1) for lk in levels_k]
        num_levels = len(level_lengths)
        L_total = sum(level_lengths)

        # Flatten levels along the time axis
        K_all = torch.cat(levels_k, dim=1)  # [B, L_total, H, D]
        V_all = torch.cat(levels_v, dim=1)  # [B, L_total, H, D]

        starts_global = torch.cat(starts_lvl, dim=0)  # [L_total]
        ends_global   = torch.cat(ends_lvl, dim=0)    # [L_total]

        # Level id per block: 0 for base tokens, 1 for first merged level, etc.
        level_ids = torch.cat([
            torch.full((L_len,), lvl, device=device, dtype=torch.long)
            for lvl, L_len in enumerate(level_lengths)
        ], dim=0)  # [L_total]

        # --- Causal base mask: block can only be used once its end token exists ---
        t_pos = torch.arange(T, device=device)   # [T]
        t_mat = t_pos.unsqueeze(1)               # [T, 1]
        ends_mat = ends_global.unsqueeze(0)      # [1, L_total]

        causal_ok = ends_mat <= t_mat           # [T, L_total]

        # --- Local window per query position ---
        # For query at t, define window [w_start(t), t], where:
        #   w_start(t) = max(0, t + 1 - local_window)
        # These tokens are "local" and must use tier-0 only.
        W = self.local_window
        if W is None or W <= 0:
            w_start = torch.zeros_like(t_pos)
        else:
            w_start = (t_pos + 1 - W).clamp(min=0)  # [T]
        w_start_mat = w_start.unsqueeze(1)          # [T, 1]

        # Blocks entirely before the local window for each t:
        #   ends_global[b] < w_start(t)
        pre_window = ends_mat < w_start_mat        # [T, L_total]

        # Distance from window start to block end (only meaningful where pre_window is true):
        #   d(t, b) = w_start(t) - ends_global[b]
        d = (w_start_mat - ends_mat).clamp(min=1)  # [T, L_total], >=1 when pre_window

        # --- Alpha gating: blocks at level i require d >= alpha^i ---
        level_ids_f = level_ids.to(torch.float32)
        # alpha^level for each block (same across all t)
        alpha_pows = (torch.ones_like(level_ids_f) * self.alpha).pow(level_ids_f)  # [L_total]
        alpha_pows_mat = alpha_pows.unsqueeze(0)                                   # [1, L_total]

        is_level0   = (level_ids == 0).unsqueeze(0)  # [1, L_total]
        is_level_ge1 = ~is_level0                    # [1, L_total]

        # Level 0 is always allowed (subject to causality).
        # For levels >=1: block must be entirely before the window AND far enough:
        #   pre_window & (d >= alpha^level)
        allowed_level = is_level0 | (pre_window & is_level_ge1 & (d >= alpha_pows_mat))

        # Final allowed mask
        use_ok = causal_ok & allowed_level          # [T, L_total]

        # --- Apply attention over flattened hierarchy ---
        q_base  = q.transpose(1, 2)                 # [B, H, T, D]
        K_all_t = K_all.transpose(1, 2)             # [B, H, L_total, D]
        V_all_t = V_all.transpose(1, 2)             # [B, H, L_total, D]

        mask_dtype = q_base.dtype
        neg_inf = torch.finfo(mask_dtype).min
        attn_mask = torch.zeros(T, L_total, device=device, dtype=mask_dtype)
        attn_mask[~use_ok] = neg_inf               # 0 for allowed, -inf for masked

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q_base, K_all_t, V_all_t,
                attn_mask=attn_mask,               # [T, L_total], broadcast over B,H
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False                    # causality is encoded in attn_mask
            )
        else:
            att = (q_base @ K_all_t.transpose(-2, -1)) * (1.0 / math.sqrt(D))  # [B, H, T, L_total]
            att = att + attn_mask.unsqueeze(0).unsqueeze(0)                    # add mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ V_all_t                                                  # [B, H, T, D]

        # Back to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        attn_classes = {
            'normal': CausalSelfAttention,
            'merge': CausalSelfAttentionMerged,
            'merge_shift': CausalSelfAttentionShifted,
            'hierarchy': CausalSelfAttentionMergedHierarchy,
        }
        self.attn = attn_classes[config.attn](config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attn: str = 'normal'

    local_window: int = 128   # number of most recent tokens kept at full resolution
    alpha: float = 2.0        # base for log distance -> level mapping
    kappa: int = 3            # maximum hierarchy level
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
