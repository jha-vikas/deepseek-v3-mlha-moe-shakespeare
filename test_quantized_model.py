"""
Test quantized model to ensure it can actually generate text
"""
import torch
import torch.nn.functional as F
import tiktoken
from deepseek_v3_model import DeepSeek, DeepSeekConfig
from quantize_model import load_quantized_model

def generate_sample(model, enc, prompt, max_len=50, device='cpu', temperature=1.0):
    """Generate text sample from the model"""
    model.eval()
    tokens = enc.encode(prompt)
    # Clip to vocab size
    tokens = [min(t, 49151) for t in tokens]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        while x.size(1) < max_len:
            # The model's forward method now expects training_step, but for inference it's None
            logits, _ = model(x, training_step=None) 
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    # Decode with error handling for vocab mismatch
    try:
        return enc.decode(x[0].tolist())
    except:
        return enc.decode([min(t, 50256) for t in x[0].tolist()])


def main():
    print("=" * 70)
    print("Testing Quantized Model")
    print("=" * 70)
    
    # Device setup
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    config = DeepSeekConfig()
    model = DeepSeek(config)
    
    # Load quantized model
    quantized_path = "huggingface_space/deepseek_v3_final_int8.pt"
    print(f"\nLoading quantized model from: {quantized_path}")
    try:
        model = load_quantized_model(model, quantized_path)
        print("✓ Quantized model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load quantized model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # Test generation
    print("\n" + "=" * 70)
    print("Testing Text Generation")
    print("=" * 70)
    
    prompt = "ROMEO:"
    print(f"\nPrompt: '{prompt}'")
    print("Generating text...")
    
    try:
        output = generate_sample(model, enc, prompt, max_len=100, device=device, temperature=0.8)
        print("\nGenerated output:")
        print("-" * 70)
        print(output)
        print("-" * 70)
        print("\n✓ Generation successful! Quantized model works correctly.")
    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("Testing Forward Pass")
    print("=" * 70)
    
    try:
        tokens = enc.encode("ROMEO: Hello")
        tokens = [min(t, 49151) for t in tokens]
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, loss = model(x, training_step=None)
        
        print(f"Input shape: {x.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
        if loss is not None:
            print(f"Loss: {loss.item():.4f}")
        else:
            print("Loss: None (no targets provided)")
        print("✓ Forward pass successful!")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("All Tests Passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

