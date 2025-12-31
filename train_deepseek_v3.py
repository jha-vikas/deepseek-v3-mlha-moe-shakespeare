"""
DeepSeek-V3 Training Script
Trains for 10000+ steps with MLA and MoE
Includes loss-less load balancing via routing bias updates
Uses corrected MLA architecture (Q not compressed, only KV compressed)
"""
import os
import math
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Import our DeepSeek-V3 model
from deepseek_v3_model import DeepSeek, DeepSeekConfig, count_parameters


# Device setup (works on both M3 Max and NVIDIA)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    # NVIDIA-specific optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # Enable TF32
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# Seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


class DataLoaderLite:
    """Simple data loader for Shakespeare text"""
    def __init__(self, B, T, file_path='input.txt'):
        self.B = B
        self.T = T
        
        # Load tokens from disk
        with open(file_path, 'r') as f:
            text = f.read()
        
        # Tokenize using GPT-2 tokenizer
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.enc = enc
        
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')
        
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        
        # Handle vocabulary size mismatch (GPT-2 tokenizer vs vocab)
        # Clip tokens to vocab size
        buf = torch.clamp(buf, 0, 49151)
        
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        
        return x, y


def get_lr(it, max_lr=7.5e-5, min_lr=7.5e-6, warmup_steps=1500, max_steps=10000):
    """
    Learning rate schedule adapted for stability
    
    DeepSeek-V3 uses max_lr=2.2e-4, but for our model with gradient amplification issues:
    - Reduced max LR to 7.5e-5 for better stability (further reduction from 1.0e-4)
    - Warmup: Linear from 0 to 7.5e-5 over 1500 steps
    - Decay: Cosine decay to 7.5e-6 (min_lr) after warmup
    
    Note: Original DeepSeek-V3 uses 2.2e-4, but our smaller model (689M) with KV compression
    requires more conservative LR to prevent overflow in bfloat16 mixed precision training.
    KV scaling (1/sqrt(num_heads)) helps but lower LR is still needed for stability.
    """
    # Linear warmup: start from 0, linearly increase to max_lr over warmup_steps
    # (DeepSeek-V3 starts from 0, not min_lr)
    if it < warmup_steps:
        warmup_ratio = (it + 1) / warmup_steps
        return warmup_ratio * max_lr  # Start from 0, not min_lr
    
    # Constant phase: maintain max_lr (DeepSeek-V3 keeps 2.2e-4 constant for most training)
    # For our shorter training (10k steps), we start cosine decay immediately after warmup
    if it > max_steps:
        return min_lr
    
    # Cosine decay after warmup (matches DeepSeek-V3's decay phase)
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def generate_sample(model, enc, prompt="ROMEO:", max_len=100, device='cpu', temperature=0.8):
    """Generate text sample from the model with temperature sampling"""
    model.eval()
    tokens = enc.encode(prompt)
    # Clip to vocab size
    tokens = [min(t, 49151) for t in tokens]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        while x.size(1) < max_len:
            logits, _ = model(x)
            
            # DEBUG 1: Check raw model logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: NaN/Inf in RAW MODEL LOGITS!")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}, Inf count: {torch.isinf(logits).sum().item()}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
                print(f"  Logits mean: {logits.mean().item():.6f}, std: {logits.std().item():.6f}")
                print(f"{'='*70}\n")
                return "ERROR: NaN/Inf in raw model logits"
            
            logits = logits[:, -1, :] / temperature  # Apply temperature
            
            # DEBUG 2: Check logits after temperature
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: NaN/Inf AFTER TEMPERATURE SCALING!")
                print(f"  Temperature: {temperature}")
                print(f"  Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
                return "ERROR: NaN/Inf after temperature"
            
            probs = F.softmax(logits, dim=-1)
            
            # DEBUG 3: Check probabilities after softmax
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: NaN/Inf/negative in SOFTMAX PROBABILITIES!")
                print(f"  Probs min: {probs.min().item():.6f}, max: {probs.max().item():.6f}")
                print(f"  Probs sum: {probs.sum().item():.6f}")
                return "ERROR: Invalid softmax probabilities"
            
            # Top-k sampling with temperature
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            # DEBUG 4: Check topk probabilities
            if torch.isnan(topk_probs).any() or torch.isinf(topk_probs).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: NaN/Inf in TOPK PROBABILITIES!")
                print(f"  Topk probs min: {topk_probs.min().item():.6f}, max: {topk_probs.max().item():.6f}")
                return "ERROR: Invalid topk probabilities"
            
            # Renormalize top-k probabilities (add epsilon to prevent division by zero)
            topk_probs_sum = topk_probs.sum(dim=-1, keepdim=True) + 1e-8
            
            # DEBUG 5: Check sum before division
            if torch.isnan(topk_probs_sum).any() or torch.isinf(topk_probs_sum).any() or (topk_probs_sum <= 0).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: Invalid TOPK_PROBS_SUM!")
                print(f"  Sum min: {topk_probs_sum.min().item():.6f}, max: {topk_probs_sum.max().item():.6f}")
                return "ERROR: Invalid topk_probs_sum"
            
            topk_probs = topk_probs / topk_probs_sum
            
            # DEBUG 6: Final check before multinomial
            if torch.isnan(topk_probs).any() or torch.isinf(topk_probs).any() or (topk_probs < 0).any():
                print(f"\n{'='*70}")
                print(f"DEBUG: Invalid probabilities BEFORE MULTINOMIAL!")
                print(f"  Topk probs min: {topk_probs.min().item():.6f}, max: {topk_probs.max().item():.6f}")
                print(f"  Topk probs sum: {topk_probs.sum().item():.6f}")
                print(f"  NaN count: {torch.isnan(topk_probs).sum().item()}")
                print(f"  Inf count: {torch.isinf(topk_probs).sum().item()}")
                print(f"  Negative count: {(topk_probs < 0).sum().item()}")
                return "ERROR: Invalid probabilities before multinomial"
            
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    model.train()
    # Decode with error handling for vocab mismatch
    try:
        return enc.decode(x[0].tolist())
    except:
        return enc.decode([min(t, 50256) for t in x[0].tolist()])


def save_checkpoint(model, optimizer, scheduler, step, loss, filepath):
    """Save complete training state"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'rng_state': torch.get_rng_state().cpu(),  # Move to CPU for saving
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath, device):
    """Load complete training state"""
    checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'rng_state' in checkpoint:
        # RNG state must be on CPU
        rng_state = checkpoint['rng_state']
        if not isinstance(rng_state, torch.ByteTensor):
            rng_state = rng_state.cpu()
        torch.set_rng_state(rng_state)
    
    # Move model to correct device after loading
    model.to(device)
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from step {checkpoint['step'] + 1}, loss: {checkpoint['loss']:.6f}")
    return checkpoint['step']


def update_load_balancing(model, x):
    """
    Update routing bias terms for load balancing (instructor's method)
    Called periodically during training
    x: token indices (B, T) - will be embedded and passed through layers
    """
    # Embed tokens first
    x = model.embed_tokens(x)  # (B, T, hidden_size)
    
    # Calculate expert load for each layer's MoE
    for layer in model.layers:
        # Pass through attention first
        x = x + layer.attention(layer.attention_norm(x))
        # Get normalized input for MoE load calculation
        x_norm = layer.ffn_norm(x)
        moe = layer.feed_forward
        expert_load = moe.calculate_expert_load(x_norm)
        # Update bias terms
        moe.update_bias_terms(expert_load)
        # Pass through MoE to get x for next layer
        x = x + moe(x_norm)


def train(resume_from_checkpoint=None, max_steps=10000):
    """Main training function"""
    
    # Track consecutive overflows - exit after 5 consecutive overflows
    consecutive_overflows = 0
    max_consecutive_overflows = 5
    
    # Initialize model
    config = DeepSeekConfig()
    model = DeepSeek(config)
    model.to(device)
    
    # Try to compile on NVIDIA (will be ignored on M3)
    if device == 'cuda':
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except:
            print("torch.compile not available, using eager mode")
    
    # Print parameter count
    param_breakdown = count_parameters(model)
    print("\nParameter Breakdown:")
    for name, count in param_breakdown.items():
        print(f"  {name}: {count:,}")
    print()
    
    # Data loader
    # Using assignment-specified batch size: B=8, T=256
    train_loader = DataLoaderLite(B=8, T=256)
    
    # Optimizer with fused option for speed
    # Initial LR will be overridden by scheduler, but set to max_lr for warmup
    # The scheduler will handle warmup from 0 to max_lr
    # Reduced max LR to 7.5e-5 for stability (further reduction from 1.0e-4)
    initial_lr = 7.5e-5  # Max LR - scheduler will start from 0 and warmup to this over 1500 steps
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=initial_lr,  # Max LR - scheduler handles warmup
            betas=(0.9, 0.95), 
            eps=1e-6,
            fused=True  # Faster on NVIDIA
        )
        print("Using fused AdamW")
    except:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=initial_lr,  # Max LR - scheduler handles warmup
            betas=(0.9, 0.95), 
            eps=1e-6
        )
        print("Using standard AdamW")
    
    # Learning rate scheduler (simple wrapper for get_lr function)
    class LRScheduler:
        def __init__(self):
            self.step_count = 0
        def step(self):
            self.step_count += 1
        def state_dict(self):
            return {'step_count': self.step_count}
        def load_state_dict(self, state):
            self.step_count = state['step_count']
    
    scheduler = LRScheduler()
    
    
    # Resume from checkpoint if provided
    start_step = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_step = load_checkpoint(model, optimizer, scheduler, resume_from_checkpoint, device)
        start_step += 1  # Start from next step
        # If resuming, set initial LR based on current step (respect warmup if still in warmup)
        if start_step > 0:
            initial_lr = get_lr(start_step - 1, max_steps=max_steps)  # Get LR for current step
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
            print(f"Resumed from step {start_step-1}, initial LR: {initial_lr:.2e}")
            print(f"  (Max LR: 7.5e-5, warmup=1500 steps)")
    
    # Training configuration
    device_type = "cuda" if device == "cuda" else "mps" if device == "mps" else "cpu"
    
    print(f"\nStarting training from step {start_step} to {max_steps}...")
    print(f"Device: {device}, Device type: {device_type}")
    print(f"Autocast (mixed precision): {'Enabled (CUDA only)' if device_type == 'cuda' else 'Disabled'}")
    print("=" * 70)
    
    # Diagnostic: Check initial model output at step 0
    if start_step == 0:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC: Checking initial model output (multiple batches)")
        print("=" * 70)
        model.eval()
        with torch.no_grad():
            # Check multiple batches to see variation
            losses = []
            for i in range(5):
                x_test, y_test = train_loader.next_batch()
                x_test, y_test = x_test.to(device), y_test.to(device)
                logits_test, loss_test = model(x_test, y_test)
                losses.append(loss_test.item())
            
            print(f"\nLoss across 5 batches:")
            for i, loss_val in enumerate(losses):
                print(f"  Batch {i}: {loss_val:.6f}")
            print(f"  Mean: {sum(losses)/len(losses):.6f}")
            print(f"  Min: {min(losses):.6f}, Max: {max(losses):.6f}")
            print(f"  Expected: {math.log(49152):.6f}")
            
            # Detailed analysis of first batch
            train_loader.current_position = 0  # Reset to start
            x_test, y_test = train_loader.next_batch()
            x_test, y_test = x_test.to(device), y_test.to(device)
            logits_test, loss_test = model(x_test, y_test)
            
            print(f"\nDetailed analysis of first training batch:")
            print(f"  Loss: {loss_test.item():.6f}")
            print(f"  Logits stats: min={logits_test.min().item():.6f}, max={logits_test.max().item():.6f}")
            print(f"  Logits mean: {logits_test.mean().item():.6f}, std: {logits_test.std().item():.6f}")
            
            # Check probability distribution
            probs_test = F.softmax(logits_test, dim=-1)
            entropy = -(probs_test * torch.log(probs_test + 1e-10)).sum(dim=-1).mean()
            print(f"  Entropy: {entropy.item():.6f} (expected: {math.log(49152):.6f})")
            print(f"  Probs max: {probs_test.max().item():.6f} (expected mean: {1.0/49152:.8f})")
            print("=" * 70 + "\n")
        
        model.train()
        # Reset data loader position for actual training
        train_loader.current_position = 0
    
    # Training loop
    for step in range(start_step, max_steps):
        t0 = time.time()
        
        # Set learning rate
        lr = get_lr(step, max_steps=max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch (non_blocking helps on CUDA, ignored on MPS)
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        # Use autocast only on CUDA (not MPS)
        if device_type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y, training_step=step)
        else:
            logits, loss = model(x, y, training_step=step)
        
        # Debug: Check first step loss discrepancy
        if step == 0:
            # Also compute loss in float32 for comparison
            model.eval()
            with torch.no_grad():
                logits_fp32, loss_fp32 = model(x, y)
            model.train()
            print(f"\n{'='*70}")
            print(f"DEBUG: Step 0 Loss Comparison")
            print(f"  Mixed precision (bfloat16) loss: {loss.item():.6f}")
            print(f"  Float32 loss: {loss_fp32.item():.6f}")
            print(f"  Difference: {abs(loss.item() - loss_fp32.item()):.6f}")
            print(f"  Expected loss: {math.log(49152):.6f}")
            print(f"{'='*70}\n")
        
        # Debug: Check logits and loss for NaN
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"\n{'='*70}")
            print(f"DEBUG: NaN/Inf in training logits at step {step}!")
            print(f"  Logits shape: {logits.shape}")
            print(f"  NaN count: {torch.isnan(logits).sum().item()}")
            print(f"  Inf count: {torch.isinf(logits).sum().item()}")
            print(f"  Logits min: {logits.min().item():.6f}, max: {logits.max().item():.6f}")
            print(f"{'='*70}\n")
            print("Stopping training to prevent further corruption.")
            print(f"Training stopped at step {step} due to NaN/Inf in logits.")
            return model  # Return early instead of break
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n{'='*70}")
            print(f"ERROR: NaN/Inf loss detected at step {step}!")
            print(f"Loss value: {loss.item()}")
            print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}")
            print(f"Logits NaN count: {torch.isnan(logits).sum().item()}")
            print(f"Logits Inf count: {torch.isinf(logits).sum().item()}")
            print(f"{'='*70}\n")
            print("Stopping training to prevent further corruption.")
            print(f"Training stopped at step {step} due to NaN/Inf loss.")
            return model  # Return early instead of break
        
        # Check for extreme logits that might cause overflow in backward pass
        logits_max = logits.abs().max().item()
        if logits_max > 50.0:  # Very large logits can cause overflow in bfloat16
            print(f"\n{'='*70}")
            print(f"WARNING: Extreme logits detected at step {step}!")
            print(f"  Logits max absolute value: {logits_max:.2f}")
            print(f"{'='*70}\n")
        
        loss.backward()
        
        # Check for overflow/NaN in gradients after backward pass
        has_overflow = False
        nan_param_names = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_overflow = True
                    nan_count = torch.isnan(param.grad).sum().item() if torch.isnan(param.grad).any() else 0
                    inf_count = torch.isinf(param.grad).sum().item() if torch.isinf(param.grad).any() else 0
                    nan_param_names.append((name, nan_count, inf_count))
        
        # If overflow detected, skip optimizer step and track consecutive overflows
        if has_overflow:
            consecutive_overflows += 1
            print(f"\n{'='*70}")
            print(f"WARNING: Overflow detected at step {step}!")
            print(f"Parameters with NaN/Inf gradients: {len(nan_param_names)}")
            if len(nan_param_names) > 0:
                for name, nan_cnt, inf_cnt in nan_param_names[:5]:  # Show first 5
                    print(f"  {name}: NaN={nan_cnt}, Inf={inf_cnt}")
                if len(nan_param_names) > 5:
                    print(f"  ... and {len(nan_param_names) - 5} more")
            
            # Check if same parameters are consistently affected
            affected_layers = set()
            for name, _, _ in nan_param_names:
                if 'layers.' in name:
                    layer_num = name.split('layers.')[1].split('.')[0]
                    affected_layers.add(layer_num)
            
            if len(affected_layers) == 1:
                print(f"  ⚠️  CONSISTENT OVERFLOW: All NaN gradients in layer {list(affected_layers)[0]}")
                print(f"  This suggests a systematic issue in this layer's attention mechanism")
            
            print(f"  Consecutive overflows: {consecutive_overflows}/{max_consecutive_overflows}")
            print(f"  Skipping optimizer step")
            
            # Reduce LR by 50% after 4th overflow, continue training
            if consecutive_overflows == 4:
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.5  # Reduce LR by 50%
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"  ⚠️  Reducing learning rate by 50% after 4th overflow")
                print(f"     LR: {current_lr:.2e} → {new_lr:.2e}")
                print(f"     Continuing training with reduced LR...")
            
            # Exit training if 5th overflow happens (even after LR reduction)
            if consecutive_overflows >= max_consecutive_overflows:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n{'='*70}")
                print(f"❌ TRAINING STOPPED: {max_consecutive_overflows} consecutive overflows detected!")
                print(f"   Last overflow at step {step}")
                print(f"   Current LR: {current_lr:.2e} (already reduced after 4th overflow)")
                print(f"   Overflow persisted even with reduced LR - training cannot continue safely.")
                print(f"{'='*70}\n")
                
                # Save final checkpoint before exiting
                final_checkpoint_path = f'deepseek_v3_checkpoint_overflow_{step}.pt'
                save_checkpoint(model, optimizer, scheduler, step, loss.item(), final_checkpoint_path)
                print(f"Final checkpoint saved to {final_checkpoint_path}")
                
                return model
            
            print(f"{'='*70}\n")
            
            # Clear gradients and skip optimizer step
            optimizer.zero_grad()
            scheduler.step()  # Still update scheduler
            # Continue to next step (try up to max_consecutive_overflows times)
            continue
        else:
            # Reset counter if no overflow
            consecutive_overflows = 0
        
        # Gradient clipping and optimizer step (only if no overflow)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Check for NaN in gradient norm (shouldn't happen if overflow check worked)
        if math.isnan(norm) or math.isinf(norm):
            print(f"\n{'='*70}")
            print(f"ERROR: NaN/Inf gradient norm detected at step {step}!")
            print(f"Gradient norm: {norm}")
            print(f"{'='*70}\n")
            
            # Run diagnostic forward pass to find where NaN originated
            print(f"\n{'='*70}")
            print(f"RUNNING DIAGNOSTIC FORWARD PASS TO FIND NaN ORIGIN...")
            print(f"{'='*70}\n")
            model.eval()
            with torch.no_grad():
                # Use same batch that caused NaN
                try:
                    logits_diag, loss_diag = model(x, y, debug_nan=True)
                    if logits_diag is None:
                        print("\n✓ Diagnostic forward pass detected NaN in forward pass!")
                        print("  NaN originated during forward pass (before backward)")
                    else:
                        print("\n✓ Diagnostic forward pass: No NaN in forward pass")
                        print("  NaN likely appeared during backward pass (gradient computation)")
                        print("  This suggests overflow in backward pass, not forward pass")
                except Exception as e:
                    print(f"Error during diagnostic forward pass: {e}")
            model.train()
            print(f"\n{'='*70}\n")
            
            # Always check for NaN in gradients (even if previous check didn't catch it)
            print("DEBUG: Checking all parameters for NaN/Inf gradients...")
            nan_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    nan_count = torch.isnan(param.grad).sum().item()
                    inf_count = torch.isinf(param.grad).sum().item()
                    if nan_count > 0 or inf_count > 0:
                        nan_params.append((name, nan_count, inf_count, param.grad.shape))
            
            if nan_params:
                print(f"Found {len(nan_params)} parameters with NaN/Inf gradients:")
                for name, nan_cnt, inf_cnt, shape in nan_params[:15]:  # Show first 15
                    print(f"  {name}: NaN={nan_cnt}, Inf={inf_cnt}, shape={shape}")
                if len(nan_params) > 15:
                    print(f"  ... and {len(nan_params) - 15} more")
            else:
                print("  No NaN/Inf found in individual parameter gradients")
                print("  NaN might be in gradient norm calculation itself")
            
            # Also check parameter values (not just gradients)
            print("\nDEBUG: Checking parameter values for NaN/Inf...")
            nan_param_values = []
            for name, param in model.named_parameters():
                nan_count = torch.isnan(param.data).sum().item()
                inf_count = torch.isinf(param.data).sum().item()
                if nan_count > 0 or inf_count > 0:
                    nan_param_values.append((name, nan_count, inf_count))
            
            if nan_param_values:
                print(f"Found {len(nan_param_values)} parameters with NaN/Inf values:")
                for name, nan_cnt, inf_cnt in nan_param_values[:10]:
                    print(f"  {name}: NaN={nan_cnt}, Inf={inf_cnt}")
                if len(nan_param_values) > 10:
                    print(f"  ... and {len(nan_param_values) - 10} more")
            else:
                print("  No NaN/Inf found in parameter values")
            
            # Check routing bias specifically (common source of issues)
            print("\nDEBUG: Checking routing bias values...")
            for name, module in model.named_modules():
                if hasattr(module, 'routing_bias') and module.routing_bias is not None:
                    bias = module.routing_bias.data
                    nan_count = torch.isnan(bias).sum().item()
                    inf_count = torch.isinf(bias).sum().item()
                    bias_min = bias.min().item() if bias.numel() > 0 else float('nan')
                    bias_max = bias.max().item() if bias.numel() > 0 else float('nan')
                    print(f"  {name}.routing_bias: NaN={nan_count}, Inf={inf_count}, min={bias_min:.4f}, max={bias_max:.4f}")
            
            print(f"\n{'='*70}")
            print("Stopping training to prevent further corruption.")
            print(f"Training stopped at step {step} due to NaN/Inf gradient norm.")
            return model  # Return early instead of break
        
        scheduler.step()
        
        # Update load balancing every 100 steps (as per assignment requirements)
        # This does an extra forward pass, but needed for proper load balancing
        if step % 100 == 0 and step > 0:
            with torch.no_grad():
                update_load_balancing(model, x)
        
        # Only synchronize when we need accurate timing (for printing/logging)
        # This avoids unnecessary sync overhead on most steps
        should_print = (step % 10 == 0) or (step % 1000 == 0) or (step == max_steps - 1)
        if should_print:
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
        
        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms
        
        # Print progress every 10 steps
        if step % 10 == 0:
            # Only call .item() when printing to avoid unnecessary CPU-GPU sync
            loss_val = loss.item()
            print(f"step {step:5d} | loss: {loss_val:.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.2f}ms")
        
        # Generate sample every 500 steps to monitor progress
        if step % 500 == 0 or step == max_steps - 1:
            print(f"\n{'='*70}")
            print(f"Sample generation at step {step}:")
            sample = generate_sample(model, train_loader.enc, "ROMEO:", max_len=150, device=device)
            print(sample)
            print(f"{'='*70}\n")
        
        # Save checkpoint every 500 steps (more frequent to catch issues early)
        # Also save at step 250 specifically (before typical NaN appearance at ~280)
        if step > 0 and (step % 500 == 0 or step == 250):
            checkpoint_path = f'deepseek_v3_checkpoint_{step}.pt'
            save_checkpoint(model, optimizer, scheduler, step, loss.item(), checkpoint_path)
            print(f"\n{'='*70}")
            print(f"CHECKPOINT SAVED AT STEP {step}")
            print(f"{'='*70}\n")
    
    # Final checkpoint (only reached if training completed normally)
    final_step = step  # Use actual final step, not max_steps - 1
    print(f"\nTraining completed normally at step {final_step}")
    print(f"Final loss: {loss.item():.6f}")
    
    # Save final model
    torch.save(model.state_dict(), "deepseek_v3_final.pt")
    print("Final model saved to deepseek_v3_final.pt")
    
    return model


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("DeepSeek-V3 Training (MLA + MoE)")
    print("=" * 70)
    
    # Parse max_steps from command line if provided
    max_steps = 10000
    resume_from = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--resume' and len(sys.argv) > 2:
            resume_from = sys.argv[2]
            print(f"\nMODE: Resuming from checkpoint {resume_from}")
        elif sys.argv[1].isdigit():
            max_steps = int(sys.argv[1])
            print(f"\nMODE: Training for {max_steps} steps")
        elif sys.argv[1] == '--resume':
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir('.') if f.startswith('deepseek_v3_checkpoint_') and f.endswith('.pt')]
            if checkpoints:
                resume_from = sorted(checkpoints, key=lambda x: int(x.split('_')[3].split('.')[0]))[-1]
                print(f"\nMODE: Resuming from latest checkpoint {resume_from}")
            else:
                print("No checkpoint found, starting fresh training")
    
    print("=" * 70)
    model = train(resume_from_checkpoint=resume_from, max_steps=max_steps)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

