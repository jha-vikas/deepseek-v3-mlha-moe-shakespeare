"""
Generate 5 outputs from trained DeepSeek-V3 model
"""
import torch
import torch.nn.functional as F
import tiktoken
from deepseek_v3_model import DeepSeek, DeepSeekConfig


def generate_sample(model, enc, prompt, max_len=200, device='cpu', temperature=0.8):
    """Generate text sample from the model"""
    model.eval()
    tokens = enc.encode(prompt)
    # Clip to vocab size
    tokens = [min(t, 49151) for t in tokens]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        while x.size(1) < max_len:
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # Renormalize top-k probabilities
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    
    # Decode with error handling for vocab mismatch
    try:
        return enc.decode(x[0].tolist())
    except:
        return enc.decode([min(t, 50256) for t in x[0].tolist()])


def main():
    # Device setup
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Load model
    config = DeepSeekConfig()
    model = DeepSeek(config)
    
    # Try to load trained weights
    try:
        model.load_state_dict(torch.load('deepseek_v3_final.pt', map_location=device))
        print("Loaded trained model weights from deepseek_v3_final.pt")
    except FileNotFoundError:
        print("Warning: deepseek_v3_final.pt not found, using random weights")
    
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding('gpt2')
    
    # Different prompts for variety (Shakespeare characters)
    prompts = [
        "ROMEO:",
        "JULIET:",
        "HAMLET:",
        "MACBETH:",
        "KING HENRY:"
    ]
    
    print("\n" + "=" * 70)
    print("Generating 5 outputs from DeepSeek-V3 model")
    print("=" * 70 + "\n")
    
    outputs = []
    for i, prompt in enumerate(prompts, 1):
        print(f"Generating output {i}/5 with prompt: '{prompt}'")
        output = generate_sample(model, enc, prompt, max_len=200, device=device, temperature=0.8)
        outputs.append({
            'prompt': prompt,
            'output': output
        })
        print(f"\nOutput {i}:")
        print(output)
        print("\n" + "-" * 70 + "\n")
    
    # Save outputs to file
    with open('output_5.txt', 'w') as f:
        f.write("DeepSeek-V3 Model - 5 Best Generated Outputs\n")
        f.write("=" * 70 + "\n\n")
        f.write("Model: DeepSeek-V3 (689M parameters)\n")
        f.write("Training: 10,000 steps, Final loss: 0.129313\n")
        f.write("=" * 70 + "\n\n")
        for i, item in enumerate(outputs, 1):
            f.write(f"Output {i} - Prompt: {item['prompt']}\n")
            f.write("-" * 70 + "\n")
            f.write(item['output'])
            f.write("\n\n" + "=" * 70 + "\n\n")
    
    print("All outputs saved to output_5.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
