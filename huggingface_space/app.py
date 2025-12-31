"""
Gradio App for DeepSeek-V3-689M Shakespeare Generator
"""
import gradio as gr
import torch
import torch.nn.functional as F
import tiktoken
import sys
import os

# Add parent directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepseek_v3_model import DeepSeek, DeepSeekConfig

# Load model
print("Loading DeepSeek-V3-689M model...")
config = DeepSeekConfig()
model = DeepSeek(config)

# Try to load quantized model first, then fallback to regular model
model_loaded = False
try:
    # Try quantized INT8 model (in same directory as app.py)
    quantized_path = os.path.join(os.path.dirname(__file__), "deepseek_v3_final_int8.pt")
    if os.path.exists(quantized_path):
        print(f"Loading quantized INT8 model from {quantized_path}...")
        checkpoint = torch.load(quantized_path, map_location=torch.device('cpu'))
        
        # Dequantize weights
        state_dict = {}
        for name, param in model.named_parameters():
            if name in checkpoint:
                quantized = checkpoint[name]
                if name + '.scale' in checkpoint:
                    scale = checkpoint[name + '.scale'].item()
                    # Dequantize: x = q * scale
                    state_dict[name] = quantized.float() * scale
                else:
                    state_dict[name] = checkpoint[name]
            else:
                state_dict[name] = checkpoint[name]
        
        model.load_state_dict(state_dict)
        print("✓ Quantized INT8 model loaded successfully!")
        model_loaded = True
except Exception as e:
    print(f"Failed to load quantized model: {e}")

# Fallback to regular float32 model
if not model_loaded:
    try:
        regular_path = os.path.join(os.path.dirname(__file__), "deepseek_v3_final.pt")
        model.load_state_dict(torch.load(regular_path, map_location=torch.device('cpu')))
        print("✓ Regular float32 model loaded successfully!")
        model_loaded = True
    except FileNotFoundError:
        print("Warning: Model checkpoint not found. Using random weights for demo.")
    except Exception as e:
        print(f"Warning: Failed to load model: {e}. Using random weights for demo.")

model.eval()


def generate_text(prompt, max_length=150, temperature=0.8):
    """Generate text from the model"""
    if not prompt:
        return "Please enter a prompt (e.g., 'ROMEO:')"
    
    enc = tiktoken.get_encoding('gpt2')
    
    try:
        tokens = enc.encode(prompt)
        # Clip to vocab size
        tokens = [min(t, 49151) for t in tokens]
        x = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            while x.size(1) < max_length:
                # Model forward expects training_step=None for inference
                logits, _ = model(x, training_step=None)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)
        
        # Decode with vocab handling
        try:
            return enc.decode(x[0].tolist())
        except:
            result_tokens = [min(t, 50256) for t in x[0].tolist()]
            return enc.decode(result_tokens)
    
    except Exception as e:
        return f"Error during generation: {str(e)}"


# Gradio Interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="e.g., ROMEO:"),
        gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Shakespeare", lines=15),
    title="DeepSeek-V3-689M Shakespeare Generator",
    description="A 689M parameter DeepSeek-V3 model trained on Shakespeare. Uses Multi-Latent Head Attention (MLHA), Mixture of Experts (MoE), RoPE, and RMSNorm.",
    examples=[
        ["ROMEO:", 150, 0.8],
        ["JULIET:", 200, 0.9],
        ["HAMLET:", 150, 0.8],
        ["MACBETH:", 200, 0.9],
        ["KING HENRY:", 150, 0.8]
    ]
)

if __name__ == "__main__":
    demo.launch()

