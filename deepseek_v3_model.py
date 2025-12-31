"""
DeepSeek-V3 Architecture: MLA + MoE
Implements Multi-Latent Attention (MLA) matching actual DeepSeek-V3 architecture
- Q: Direct projection (no compression) - eliminates gradient amplification
- KV: Compressed to latent space (only KV compressed)
- True RoPE, RMSNorm, MoE with loss-less load balancing
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek model with instructor's parameters"""
    hidden_size: int = 576              # Hidden dimension (scaled to match SmolLM2, was 768)
    n_layers: int = 30                  # Number of transformer layers
    n_heads: int = 16                   # Number of attention heads (576/16=36 divides evenly)
    vocab_size: int = 49152             # Vocabulary size
    intermediate_size: int = 1536       # MLP hidden dimension
    max_seq_len: int = 2048             # Maximum sequence length (instructor's param)
    compression_ratio: int = 8          # MLA compression ratio (instructor's param)
    num_experts: int = 8                # Total number of experts (instructor's param)
    num_shared_experts: int = 1         # Number of shared (always-active) experts
    top_k_experts: int = 2              # Top-k experts per token (instructor's param)
    norm_eps: float = 1e-4              # RMSNorm epsilon
    
    @property
    def latent_dim(self):
        """Latent dimension for MLA compression (KV only)"""
        return self.hidden_size // self.compression_ratio  # 576 / 8 = 72
    
    @property
    def head_dim(self):
        """Head dimension for attention"""
        # 576 / 16 = 36, which divides evenly
        return self.hidden_size // self.n_heads  # 576 // 16 = 36
    
    @property
    def num_routed_experts(self):
        """Number of experts that can be routed to (excluding shared)"""
        return self.num_experts - self.num_shared_experts  # 8 - 1 = 7


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    More stable and efficient than LayerNorm for deep models
    """
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: (B, T, dim)
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class RoPE(nn.Module):
    """
    Rotary Position Embeddings
    Applies rotation to Q and K based on position
    Zero parameters - pure rotation!
    Matches actual DeepSeek-V3 RoPE implementation
    """
    def __init__(self, head_dim, max_seq_len=2048, theta=100000.0):
        super().__init__()
        self.head_dim = head_dim
        
        # Precompute frequencies - standard Llama approach
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, start_pos=0):
        """
        Apply rotary embeddings to x using the standard Llama approach
        x: (B, n_heads, T, head_dim)
        """
        B, n_heads, T, head_dim = x.shape
        
        # Get cos and sin for this sequence
        cos = self.cos_cached[start_pos:start_pos + T]  # (T, head_dim)
        sin = self.sin_cached[start_pos:start_pos + T]  # (T, head_dim)
        
        # Reshape for broadcasting: (1, 1, T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Rotate using the real-valued formula
        # x_rot = [x1*cos - x2*sin, x1*sin + x2*cos]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        
        rotated = torch.cat([
            x1 * cos[..., : head_dim // 2] - x2 * sin[..., head_dim // 2 :],
            x1 * sin[..., : head_dim // 2] + x2 * cos[..., head_dim // 2 :]
        ], dim=-1)
        
        return rotated


class MultiLatentHeadAttention(nn.Module):
    """
    Multi-Latent Head Attention (MLA) matching actual DeepSeek-V3 architecture
    - Q: Direct projection from hidden_size to num_heads * head_dim (NO compression)
    - KV: Compressed to latent space, then expanded per-head
    - True RoPE rotation (zero parameters, matches actual DeepSeek)
    - KV scaling: Still needed for KV compression to prevent gradient amplification
      (Q doesn't need scaling since it's direct, but KV compression still causes amplification)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_heads
        self.latent_dim = config.latent_dim
        self.head_dim = config.head_dim
        
        # Q: Direct projection (no compression) - matches actual DeepSeek-V3
        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * self.head_dim, bias=False)
        
        # KV: Compress to latent space (shared compression)
        self.kv_proj_d = nn.Linear(config.hidden_size, config.latent_dim, bias=False)
        
        # Per-head up-projections for K (from latent to head_dim)
        self.k_up = nn.ModuleList([
            nn.Linear(config.latent_dim, self.head_dim, bias=False) 
            for _ in range(config.n_heads)
        ])
        
        # Per-head up-projections for V
        self.v_up = nn.ModuleList([
            nn.Linear(config.latent_dim, self.head_dim, bias=False) 
            for _ in range(config.n_heads)
        ])
        
        # True RoPE rotation (zero parameters, matches actual DeepSeek)
        self.rope = RoPE(self.head_dim, max_seq_len=config.max_seq_len)
        
        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Mark output projection for scaled initialization
        self.o_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x, training_step=None):
        B, T, C = x.shape
        
        # Q: Direct projection (no compression) - no gradient amplification
        q = self.q_proj(x)  # (B, T, num_heads * head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        
        # KV: Compress to latent space, then expand per-head
        kv_latent = self.kv_proj_d(x)  # (B, T, latent_dim)
        
        # CRITICAL: Scale down KV latent to prevent gradient amplification
        # During backward, gradients from all heads accumulate in kv_latent
        # This creates num_heads x gradient amplification, causing overflow in bfloat16
        # Solution: Scale down by 1/num_heads to eliminate amplification
        # Q doesn't need scaling (direct projection), but KV still does (compressed)
        scale_factor = 1.0 / math.sqrt(self.num_heads)
        kv_latent = kv_latent * scale_factor
        
        # Expand per head for K and V
        k_list = [self.k_up[h](kv_latent) for h in range(self.num_heads)]  # List of (B, T, head_dim)
        v_list = [self.v_up[h](kv_latent) for h in range(self.num_heads)]  # List of (B, T, head_dim)
        
        k = torch.stack(k_list, dim=1)  # (B, num_heads, T, head_dim)
        v = torch.stack(v_list, dim=1)  # (B, num_heads, T, head_dim)
        
        # Apply true RoPE rotation (in-place, zero parameters)
        q = self.rope(q)  # (B, num_heads, T, head_dim) - same shape, rotated
        k = self.rope(k)  # (B, num_heads, T, head_dim) - same shape, rotated
        
        # Clamp Q and K values only during first 20 steps to prevent overflow
        if training_step is not None and training_step < 20:
            q = torch.clamp(q, -50, 50)
            k = torch.clamp(k, -50, 50)
        
        # Flash Attention: Use PyTorch's optimized scaled_dot_product_attention
        # q and k have same dimension (head_dim) after rotation
        try:
            # Use flash attention (PyTorch 2.0+) - automatically optimized
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,  # Causal masking: prevent seeing future tokens
                dropout_p=0.0  # No dropout during training
            )
        except (AttributeError, RuntimeError):
            # Fallback to standard attention if flash attention not available
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Apply causal mask: prevent seeing future tokens
            # attn shape: (B, num_heads, T, T)
            causal_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            # Clamp attention scores to prevent numerical issues
            attn = torch.clamp(attn, min=-1e9, max=1e9)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
        
        # Store attention output before projection for NaN check
        attn_out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Assert no NaNs immediately after attention
        assert torch.isfinite(attn_out).all(), "NaN/Inf detected in attention output"
        
        # Output projection
        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network
    Used as individual expert in MoE
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # Down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # Up
        
        # Mark down projection for scaled initialization
        self.w2.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        # SwiGLU: swish(W1(x)) * W3(x), then project down with W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts with Loss-less Load Balancing
    - 1 shared expert (always active)
    - 7 switched experts (routed)
    - Top-k = 2 (shared + 1 switched)
    - Load balancing via routing bias updates (no auxiliary loss)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.top_k = config.top_k_experts
        
        # Router: decides which switched experts to use (excludes shared expert)
        self.router = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        
        # Routing bias for load balancing (instructor's method)
        self.routing_bias = nn.Parameter(torch.zeros(config.num_routed_experts))
        
        # Shared expert (always active)
        self.shared_expert = SwiGLU(config.hidden_size, config.intermediate_size)
        
        # Switched experts
        self.switched_experts = nn.ModuleList([
            SwiGLU(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_routed_experts)
        ])
        
        # Mark shared expert's down projection for scaled initialization
        self.shared_expert.w2.NANOGPT_SCALE_INIT = 1
        for expert in self.switched_experts:
            expert.w2.NANOGPT_SCALE_INIT = 1
    
    def update_bias_terms(self, expert_load):
        """
        Update routing bias terms based on expert load (instructor's method)
        This is the loss-less load balancing mechanism
        
        Args:
            expert_load: (num_routed_experts,) tensor with normalized load per expert
        """
        # Target load: uniform distribution
        target_load = 1.0 / self.num_routed_experts
        
        # Load difference: positive = overloaded, negative = underloaded
        load_diff = expert_load - target_load
        
        # Dynamic update rate based on magnitude of imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        
        # Update routing bias: decrease for overloaded, increase for underloaded
        # This makes overloaded experts less likely and underloaded more likely
        self.routing_bias.data -= update_rate * load_diff
        
        # Clamp routing bias to prevent extreme values (numerical stability)
        self.routing_bias.data.clamp_(-5, 5)
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Shared expert always processes all tokens
        shared_output = self.shared_expert(x)  # (B, T, C)
        
        # Router decides which switched experts to use
        router_logits = self.router(x)  # (B, T, num_routed_experts)
        router_logits = router_logits + self.routing_bias  # Add bias for load balancing
        
        # Clamp router_logits to prevent extreme values (numerical stability)
        router_logits = torch.clamp(router_logits, min=-50.0, max=50.0)
        
        # Convert to probabilities using sigmoid (instructor's approach)
        router_probs = torch.sigmoid(router_logits)  # (B, T, num_routed_experts)
        
        # Check for NaN/Inf in router_probs
        if torch.isnan(router_probs).any() or torch.isinf(router_probs).any():
            print(f"ERROR: NaN/Inf in router_probs! router_logits min={router_logits.min().item():.6f}, max={router_logits.max().item():.6f}")
            router_probs = torch.clamp(router_probs, min=1e-8, max=1.0-1e-8)
        
        # Select top-k switched experts (k = top_k - 1, since shared is always active)
        num_switched = self.top_k - 1  # 2 - 1 = 1
        topk_probs, topk_indices = torch.topk(router_probs, num_switched, dim=-1)  # (B, T, 1)
        
        # Clamp topk_probs to ensure numerical stability
        topk_probs = torch.clamp(topk_probs, min=1e-8, max=1.0-1e-8)
        
        # Optimized: Use scatter/gather for better GPU utilization
        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, C)  # (B*T, C)
        topk_indices_flat = topk_indices.view(-1, num_switched)  # (B*T, 1)
        topk_probs_flat = topk_probs.view(-1, num_switched)  # (B*T, 1)
        
        # Get the selected expert ID and probability for each token
        selected_expert = topk_indices_flat[:, 0]  # (B*T,)
        selected_prob = topk_probs_flat[:, 0]  # (B*T,)
        
        # Parallel MoE: Process all experts simultaneously for better GPU utilization
        # Group tokens by expert first, then process all experts in parallel
        switched_output_flat = torch.zeros_like(x_flat)  # (B*T, C)
        
        # Prepare expert inputs (group tokens by expert)
        expert_tasks = []  # List of (expert_id, indices, input_tensor, weights)
        for expert_id in range(self.num_routed_experts):
            mask = (selected_expert == expert_id)  # (B*T,)
            if mask.any():
                indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # (num_selected,)
                expert_input = x_flat[indices]  # (num_selected, C)
                expert_weights = selected_prob[indices]  # (num_selected,)
                expert_tasks.append((expert_id, indices, expert_input, expert_weights))
        
        # Process all experts in parallel (PyTorch will schedule independent operations concurrently)
        # Each expert forward() call is independent and can run in parallel
        expert_outputs = []
        for expert_id, indices, expert_input, expert_weights in expert_tasks:
            # Process expert (this can run in parallel with other experts)
            expert_output = self.switched_experts[expert_id](expert_input)  # (num_selected, C)
            # Apply probability weights
            expert_output = expert_output * expert_weights.unsqueeze(-1)  # (num_selected, C)
            expert_outputs.append((indices, expert_output))
        
        # Scatter results back to original positions
        for indices, expert_output in expert_outputs:
            switched_output_flat[indices] = expert_output
        
        # Reshape back
        switched_output = switched_output_flat.view(B, T, C)  # (B, T, C)
        
        # Combine shared and switched experts
        # Shared gets 0.5 weight, switched gets weighted by router probabilities
        total_weight = 0.5 + topk_probs.sum(dim=-1)  # (B, T)
        # Add epsilon to prevent division by zero (numerical stability)
        total_weight = total_weight.unsqueeze(-1) + 1e-8  # (B, T, 1)
        
        # Clamp total_weight to prevent extreme values
        total_weight = torch.clamp(total_weight, min=0.5, max=2.0)
        
        output = (0.5 / total_weight) * shared_output + \
                 (switched_output / total_weight)
        
        # Final check for NaN/Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"ERROR: NaN/Inf in MoE output! shared_output stats: min={shared_output.min().item():.6f}, max={shared_output.max().item():.6f}")
            print(f"  switched_output stats: min={switched_output.min().item():.6f}, max={switched_output.max().item():.6f}")
            print(f"  total_weight stats: min={total_weight.min().item():.6f}, max={total_weight.max().item():.6f}")
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output
    
    def calculate_expert_load(self, x):
        """
        Calculate expert load for load balancing (called during training)
        Returns normalized load per expert
        Optimized using scatter_add_ for better GPU utilization
        """
        B, T, C = x.shape
        
        # Get routing decisions
        router_logits = self.router(x) + self.routing_bias
        router_probs = torch.sigmoid(router_logits)
        num_switched = self.top_k - 1
        _, topk_indices = torch.topk(router_probs, num_switched, dim=-1)
        
        # Optimized: Use scatter_add_ instead of nested loops
        expert_load = torch.zeros(self.num_routed_experts, device=x.device)
        
        # Flatten indices and use scatter_add_ for efficient counting
        topk_indices_flat = topk_indices.view(-1)  # (B*T*num_switched,)
        expert_load.scatter_add_(0, topk_indices_flat, torch.ones_like(topk_indices_flat, dtype=torch.float))
        
        # Normalize by total tokens * top_k (add epsilon for numerical stability)
        expert_load = expert_load / (B * T * self.top_k + 1e-8)
        
        return expert_load


class TransformerBlock(nn.Module):
    """
    Single transformer block with Pre-RMSNorm architecture
    Uses MLA and MoE
    Matches actual DeepSeek-V3 block structure
    """
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attention = MultiLatentHeadAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.feed_forward = MixtureOfExperts(config)
    
    def forward(self, x, training_step=None):
        # Attention with residual (pre-norm architecture)
        x = x + self.attention(self.attention_norm(x), training_step=training_step)
        # Feed-forward with residual (pre-norm architecture)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class DeepSeek(nn.Module):
    """
    DeepSeek-V3 Model with MLA and MoE
    Matches actual DeepSeek-V3 architecture
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final normalization (before LM head)
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        
        # Output head (tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following Llama/GPT-2 style"""
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down residual projections for stability
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, debug_nan=False, training_step=None):
        """
        Forward pass
        idx: (B, T) token indices
        targets: (B, T) target indices for loss calculation
        debug_nan: If True, check for NaN at each layer and print diagnostics
        training_step: Current training step (for step-based clamping)
        """
        B, T = idx.size()
        
        # Token embeddings
        x = self.embed_tokens(idx)  # (B, T, hidden_size)
        
        if debug_nan:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"\n{'='*70}")
                print(f"NaN/Inf detected in EMBEDDINGS output!")
                print(f"  NaN count: {torch.isnan(x).sum().item()}")
                print(f"  Inf count: {torch.isinf(x).sum().item()}")
                print(f"  Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
                print(f"{'='*70}\n")
                return None, None
        
        # Pass through all transformer blocks
        for layer_idx, layer in enumerate(self.layers):
            x_before = x.clone()
            
            # Check attention output
            attn_norm_out = layer.attention_norm(x)
            if debug_nan:
                if torch.isnan(attn_norm_out).any() or torch.isinf(attn_norm_out).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - ATTENTION NORM!")
                    print(f"  NaN count: {torch.isnan(attn_norm_out).sum().item()}")
                    print(f"  Inf count: {torch.isinf(attn_norm_out).sum().item()}")
                    print(f"{'='*70}\n")
                    return None, None
            
            attn_out = layer.attention(attn_norm_out, training_step=training_step)
            if debug_nan:
                if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - ATTENTION OUTPUT!")
                    print(f"  NaN count: {torch.isnan(attn_out).sum().item()}")
                    print(f"  Inf count: {torch.isinf(attn_out).sum().item()}")
                    print(f"  Min: {attn_out.min().item():.6f}, Max: {attn_out.max().item():.6f}")
                    print(f"{'='*70}\n")
                    return None, None
            
            x = x + attn_out  # Residual connection
            
            if debug_nan:
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - AFTER ATTENTION RESIDUAL!")
                    print(f"  NaN count: {torch.isnan(x).sum().item()}")
                    print(f"  Inf count: {torch.isinf(x).sum().item()}")
                    print(f"{'='*70}\n")
                    return None, None
            
            # Check MoE output
            ffn_norm_out = layer.ffn_norm(x)
            if debug_nan:
                if torch.isnan(ffn_norm_out).any() or torch.isinf(ffn_norm_out).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - FFN NORM!")
                    print(f"  NaN count: {torch.isnan(ffn_norm_out).sum().item()}")
                    print(f"  Inf count: {torch.isinf(ffn_norm_out).sum().item()}")
                    print(f"{'='*70}\n")
                    return None, None
            
            moe_out = layer.feed_forward(ffn_norm_out)
            if debug_nan:
                if torch.isnan(moe_out).any() or torch.isinf(moe_out).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - MoE OUTPUT!")
                    print(f"  NaN count: {torch.isnan(moe_out).sum().item()}")
                    print(f"  Inf count: {torch.isinf(moe_out).sum().item()}")
                    print(f"  Min: {moe_out.min().item():.6f}, Max: {moe_out.max().item():.6f}")
                    print(f"{'='*70}\n")
                    return None, None
            
            x = x + moe_out  # Residual connection
            
            if debug_nan:
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"\n{'='*70}")
                    print(f"NaN/Inf detected in LAYER {layer_idx} - AFTER MoE RESIDUAL!")
                    print(f"  NaN count: {torch.isnan(x).sum().item()}")
                    print(f"  Inf count: {torch.isinf(x).sum().item()}")
                    print(f"  Min: {x.min().item():.6f}, Max: {x.max().item():.6f}")
                    print(f"{'='*70}\n")
                    return None, None
        
        # Final normalization
        x = self.norm(x)
        
        if debug_nan:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"\n{'='*70}")
                print(f"NaN/Inf detected in FINAL NORM!")
                print(f"  NaN count: {torch.isnan(x).sum().item()}")
                print(f"  Inf count: {torch.isinf(x).sum().item()}")
                print(f"{'='*70}\n")
                return None, None
        
        # Output logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if debug_nan:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n{'='*70}")
                print(f"NaN/Inf detected in LOGITS!")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                print(f"  Inf count: {torch.isinf(logits).sum().item()}")
                print(f"{'='*70}\n")
                return None, None
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


def count_parameters(model):
    """Calculate total parameters in the model"""
    total = sum(p.numel() for p in model.parameters())
    
    # Breakdown by component
    breakdown = {}
    breakdown['Token Embeddings'] = model.embed_tokens.weight.numel()
    
    # Single layer
    if len(model.layers) > 0:
        layer = model.layers[0]
        layer_params = sum(p.numel() for p in layer.parameters())
        breakdown['Per Layer'] = layer_params
        breakdown['All Layers'] = layer_params * len(model.layers)
    
    breakdown['Final Norm'] = model.norm.weight.numel()
    breakdown['LM Head'] = 0  # Tied with embeddings
    breakdown['Total'] = total
    
    return breakdown


if __name__ == "__main__":
    # Test the model
    config = DeepSeekConfig()
    model = DeepSeek(config)
    
    print("DeepSeek-V3 Architecture")
    print("=" * 50)
    print(f"Layers: {config.n_layers}")
    print(f"Hidden Dim: {config.hidden_size}")
    print(f"Heads: {config.n_heads}")
    print(f"Vocab: {config.vocab_size}")
    print(f"Compression Ratio: {config.compression_ratio} (latent_dim={config.latent_dim})")
    print(f"Experts: {config.num_experts} (shared={config.num_shared_experts}, routed={config.num_routed_experts})")
    print(f"Top-k: {config.top_k_experts}")
    print()
    
    # Count parameters
    breakdown = count_parameters(model)
    print("Parameter Breakdown:")
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")
    print()
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 128))  # (B=2, T=128)
    y = torch.randint(0, config.vocab_size, (2, 128))
    
    logits, loss = model(x, y)
    print(f"Forward pass test:")
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f} (should be ~ln({config.vocab_size}) = {math.log(config.vocab_size):.4f})")

