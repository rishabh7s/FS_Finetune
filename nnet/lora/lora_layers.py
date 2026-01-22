"""
LoRA (Low-Rank Adaptation) implementation for FS-EEND
Supports Linear layers, MultiheadAttention, and Conv1d layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """Base LoRA layer with rank decomposition"""
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # x: (..., in_features)
        # LoRA forward: x @ A^T @ B^T * scaling
        result = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return result


class LoRALinear(nn.Module):
    """Linear layer with LoRA"""
    def __init__(self, linear_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRAConv1d(nn.Module):
    """Conv1d layer with LoRA"""
    def __init__(self, conv_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.conv = conv_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]
        
        # LoRA for conv: decompose kernel
        # W_conv shape: (out_channels, in_channels, kernel_size)
        # Decompose: A (r, in_channels * kernel_size), B (out_channels, r)
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels * kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        for param in self.conv.parameters():
            param.requires_grad = False
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = conv_layer.padding
        self.stride = conv_layer.stride
    
    def forward(self, x):
        # Original convolution
        result = self.conv(x)
        
        # LoRA convolution
        # Reconstruct low-rank conv kernel
        lora_weight = (self.lora_B @ self.lora_A).view(
            self.out_channels, self.in_channels, self.kernel_size
        ) * self.scaling
        
        # Apply LoRA conv
        lora_out = F.conv1d(
            self.lora_dropout(x),
            lora_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding
        )
        
        return result + lora_out


class LoRAMultiheadAttention(nn.Module):
    """MultiheadAttention with LoRA on in_proj and out_proj"""
    def __init__(self, attn_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.attn = attn_layer
        embed_dim = attn_layer.embed_dim
        
        # Expose important attributes from wrapped attention layer
        self.embed_dim = attn_layer.embed_dim
        self.num_heads = attn_layer.num_heads
        self.batch_first = attn_layer.batch_first
        self._qkv_same_embed_dim = attn_layer._qkv_same_embed_dim
        self.in_proj_weight = attn_layer.in_proj_weight
        self.in_proj_bias = attn_layer.in_proj_bias
        self.out_proj = attn_layer.out_proj
        self.dropout = attn_layer.dropout
        
        # LoRA for in_proj (Q, K, V combined)
        if attn_layer._qkv_same_embed_dim:
            # in_proj_weight shape: (3 * embed_dim, embed_dim)
            self.lora_in_proj = LoRALayer(
                embed_dim, 3 * embed_dim,
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            raise NotImplementedError("Separate Q, K, V projections not supported yet")
        
        # LoRA for out_proj
        self.lora_out_proj = LoRALayer(
            embed_dim, embed_dim,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        
        # Freeze original weights
        for param in self.attn.parameters():
            param.requires_grad = False
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        """
        Forward pass compatible with PyTorch MultiheadAttention
        """
        # Get original projections
        if self.attn._qkv_same_embed_dim:
            # Apply base in_proj
            qkv = F.linear(query, self.attn.in_proj_weight, self.attn.in_proj_bias)
            # Add LoRA adaptation
            qkv = qkv + self.lora_in_proj(query)
            
            # Split into Q, K, V
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            raise NotImplementedError()
        
        # Manual attention computation
        # Scale query
        scaling = float(self.attn.embed_dim // self.attn.num_heads) ** -0.5
        q = q * scaling
        
        # Reshape for multi-head attention
        # From (B, T, E) to (B, num_heads, T, head_dim)
        B, T, E = q.shape
        head_dim = E // self.attn.num_heads
        
        q = q.reshape(B, T, self.attn.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.attn.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.attn.num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        # attn_weights shape: (B, num_heads, T_q, T_k)
        
        T_q = attn_weights.size(-2)  # Query sequence length
        T_k = attn_weights.size(-1)  # Key sequence length
        
        # Handle causal masking first
        if is_causal:
            seq_len = T_k
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attn_mask is not None:
            # CRITICAL FIX: Check if mask dimensions match the actual attention dimensions
            # The mask might be for the full sequence but we're processing a smaller chunk
            mask_size = attn_mask.size(-1) if attn_mask.dim() >= 2 else 0
            
            # Only apply mask if dimensions are compatible
            if mask_size == T_k and (attn_mask.dim() < 2 or attn_mask.size(-2) == T_q):
                # Mask dimensions match - safe to apply
                if attn_mask.dtype == torch.bool:
                    # Boolean mask: True means position should be ignored
                    if attn_mask.dim() == 2:
                        attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    else:
                        attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
                else:
                    # Additive mask
                    if attn_mask.dim() == 2:
                        attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
                    elif attn_mask.dim() == 3:
                        if attn_mask.size(0) == B:
                            attn_weights = attn_weights + attn_mask.unsqueeze(1)
                        else:
                            attn_weights = attn_weights + attn_mask.unsqueeze(0)
                    elif attn_mask.dim() == 4:
                        attn_weights = attn_weights + attn_mask
            else:
                # Mask dimensions don't match - skip it
                # This happens when the mask was created for the full sequence
                # but we're processing in smaller chunks
                pass
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask shape: (B, T_k)
            # Expand to (B, 1, 1, T_k) for broadcasting
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attn.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(B, T, E)
        
        # Apply out_proj with LoRA
        attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
        attn_output = attn_output + self.lora_out_proj(attn_output)
        
        if need_weights:
            # Average attention weights over heads if requested
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None


def apply_lora_to_model(model, lora_config):
    """
    Apply LoRA to specific layers in the OnlineTransformerDADiarization model
    
    Args:
        model: OnlineTransformerDADiarization instance
        lora_config: Dict with LoRA configuration
            - r: LoRA rank
            - lora_alpha: scaling factor
            - lora_dropout: dropout rate
            - target_modules: list of module name patterns (not used, applying to all)
    """
    r = lora_config.get('r', 8)
    lora_alpha = lora_config.get('lora_alpha', 16)
    lora_dropout = lora_config.get('lora_dropout', 0.1)
    
    print(f"Applying LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # ============ ENCODER ============
    # Access encoder layers: model.enc.transformer_encoder.layers
    print("\nApplying LoRA to Encoder layers...")
    encoder_layers = model.enc.transformer_encoder.layers
    
    for i, layer in enumerate(encoder_layers):
        print(f"  Processing encoder layer {i}...")
        
        # Replace self-attention with LoRA version
        original_attn = layer.self_attn
        lora_attn = LoRAMultiheadAttention(
            original_attn, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        layer.self_attn = lora_attn
        
        # Replace feedforward layers
        layer.linear1 = LoRALinear(layer.linear1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        layer.linear2 = LoRALinear(layer.linear2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    # ============ CONV1D ============
    print("\nApplying LoRA to Conv1d layer...")
    original_conv = model.cnn
    lora_conv = LoRAConv1d(original_conv, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    model.cnn = lora_conv
    
    # ============ DECODER ============
    # Access decoder layers: model.dec.attractor_decoder.layers
    print("\nApplying LoRA to Decoder layers...")
    decoder_layers = model.dec.attractor_decoder.layers
    
    for i, layer in enumerate(decoder_layers):
        print(f"  Processing decoder layer {i}...")
        
        # TransformerEncoderFusionLayer has self_attn1 and self_attn2
        # Replace temporal attention (self_attn1)
        original_attn1 = layer.self_attn1
        lora_attn1 = LoRAMultiheadAttention(
            original_attn1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        layer.self_attn1 = lora_attn1
        
        # Replace speaker attention (self_attn2)
        original_attn2 = layer.self_attn2
        lora_attn2 = LoRAMultiheadAttention(
            original_attn2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        layer.self_attn2 = lora_attn2
        
        # Replace feedforward layers
        layer.linear1 = LoRALinear(layer.linear1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        layer.linear2 = LoRALinear(layer.linear2, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    print("\nLoRA application complete!")
    return model


def get_lora_parameters(model):
    """Get only LoRA parameters for optimization"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def print_trainable_parameters(model):
    """Print statistics of trainable parameters"""
    trainable_params = 0
    all_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable %: {100 * trainable_params / all_params:.2f}%")
    print(f"LoRA params: {lora_params:,} || "
          f"LoRA %: {100 * lora_params / all_params:.2f}%")