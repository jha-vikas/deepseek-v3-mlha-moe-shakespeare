"""
Quantize DeepSeek-V3 model to INT8 for HuggingFace Space deployment
Reduces model size from ~2.6 GB (float32) to ~0.69 GB (INT8)
"""
import torch
import torch.nn as nn
from deepseek_v3_model import DeepSeek, DeepSeekConfig
import os


def quantize_model_to_int8(model, checkpoint_path, output_path):
    """
    Quantize model weights to INT8 using symmetric quantization
    
    Args:
        model: DeepSeek model instance
        checkpoint_path: Path to original float32 checkpoint
        output_path: Path to save quantized checkpoint
    """
    print("=" * 70)
    print("DeepSeek-V3 INT8 Quantization")
    print("=" * 70)
    
    # Load original checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("✓ Checkpoint loaded successfully")
    
    # Get original size
    original_size = os.path.getsize(checkpoint_path) / (1024 ** 3)  # GB
    print(f"Original checkpoint size: {original_size:.2f} GB")
    
    # Quantize model weights to INT8
    print("\nQuantizing model weights to INT8...")
    print("Using symmetric quantization (zero_point=0)")
    
    # Get model's parameter names to identify what to quantize
    param_names = set(name for name, _ in model.named_parameters())
    
    # Create quantized state dict
    quantized_state_dict = {}
    total_params = 0
    quantized_params = 0
    
    # Process all items in checkpoint (parameters + buffers)
    for name, tensor in checkpoint.items():
        # Check if this is a parameter (trainable) that needs quantization
        is_parameter = name in param_names
        
        if is_parameter and tensor.dtype == torch.float32:
            total_params += tensor.numel()
            
            # Symmetric quantization: q = round(x / scale)
            # Scale = max(|min|, |max|) / 127
            param_abs_max = torch.abs(tensor).max().item()
            
            if param_abs_max == 0:
                # Zero tensor
                scale = 1.0
                param_quantized = torch.zeros_like(tensor, dtype=torch.int8)
            else:
                # Calculate scale for symmetric quantization
                scale = param_abs_max / 127.0
                
                # Quantize: q = round(x / scale), clamp to [-128, 127]
                param_quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            
            # Store quantized tensor and scale
            quantized_state_dict[name] = param_quantized
            quantized_state_dict[name + '.scale'] = torch.tensor(scale, dtype=torch.float32)
            
            quantized_params += tensor.numel()
            
            if len(quantized_state_dict) % 50 == 0:
                print(f"  Processed {len(quantized_state_dict)//2} parameters...")
        else:
            # Keep buffers and non-float32 parameters as-is
            quantized_state_dict[name] = tensor
    
    # Save quantized checkpoint
    print(f"\nSaving quantized checkpoint to: {output_path}")
    torch.save(quantized_state_dict, output_path)
    
    # Get quantized size
    quantized_size = os.path.getsize(output_path) / (1024 ** 3)  # GB
    reduction = (1 - quantized_size / original_size) * 100
    
    print("\n" + "=" * 70)
    print("Quantization Complete!")
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Quantized parameters: {quantized_params:,}")
    print(f"Original size:  {original_size:.2f} GB")
    print(f"Quantized size: {quantized_size:.2f} GB")
    print(f"Reduction:      {reduction:.1f}%")
    print(f"\nQuantized model saved to: {output_path}")
    print("=" * 70)


def load_quantized_model(model, checkpoint_path):
    """
    Load quantized INT8 model and dequantize weights
    
    Args:
        model: DeepSeek model instance (initialized)
        checkpoint_path: Path to quantized checkpoint
    """
    print(f"Loading quantized checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Dequantize and load weights
    state_dict = {}
    
    # Process all items in checkpoint
    for name, tensor in checkpoint.items():
        # Skip scale entries (they're metadata, not model weights)
        if name.endswith('.scale'):
            continue
            
        # Check if this is a quantized parameter (has scale)
        if name + '.scale' in checkpoint:
            # Dequantize: x = q * scale
            scale = checkpoint[name + '.scale'].item()
            dequantized = tensor.float() * scale
            state_dict[name] = dequantized
        else:
            # Keep as-is (buffers, non-quantized parameters)
            state_dict[name] = tensor
    
    # Load the state dict (with strict=False to handle tied weights like lm_head.weight)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        # Filter out expected missing keys (tied weights)
        expected_missing = [k for k in missing_keys if 'lm_head.weight' in k]
        if expected_missing:
            print(f"  Note: lm_head.weight is tied with embed_tokens.weight (expected)")
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]
        if unexpected_missing:
            print(f"  Warning: Missing keys: {len(unexpected_missing)} keys")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys: {len(unexpected_keys)} keys")
    
    print("✓ Quantized model loaded and dequantized")
    return model


def main():
    """Main function to quantize the model"""
    # Paths
    checkpoint_path = "huggingface_space/deepseek_v3_final.pt"
    output_path = "huggingface_space/deepseek_v3_final_int8.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Initialize model
    print("Initializing model...")
    config = DeepSeekConfig()
    model = DeepSeek(config)
    model.eval()
    
    # Quantize
    quantize_model_to_int8(model, checkpoint_path, output_path)
    
    # Verify loading works
    print("\nVerifying quantized checkpoint can be loaded...")
    model_test = DeepSeek(config)
    load_quantized_model(model_test, output_path)
    print("✓ Loading verification successful!")
    
    # Test actual inference
    print("\nTesting inference with quantized model...")
    try:
        import tiktoken
        import torch.nn.functional as F
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("ROMEO:")
        tokens = [min(t, 49151) for t in tokens]
        x = torch.tensor(tokens).unsqueeze(0)
        
        model_test.eval()
        with torch.no_grad():
            logits, _ = model_test(x, training_step=None)
        
        # Check for NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("✗ Warning: NaN/Inf detected in logits!")
        else:
            print(f"✓ Inference test passed! Logits shape: {logits.shape}, stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}")
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Quantization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

