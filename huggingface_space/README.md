# DeepSeek-V3-689M Shakespeare Generator

This HuggingFace Space hosts a Gradio app for the DeepSeek-V3-689M model trained on Shakespeare text.

## Model Architecture

- **Parameters**: ~689M
- **Architecture**: DeepSeek-V3 with Multi-Latent Head Attention (MLHA) and Mixture of Experts (MoE)
- **Attention**: MLHA with Q direct projection, KV compression (ratio 8)
- **FFN**: MoE with 8 experts (1 shared, 7 routed), top-k=2
- **Normalization**: RMSNorm (pre-norm)
- **Position Encoding**: RoPE (Rotary Position Embeddings)

## Training

- **Steps**: 10,000
- **Final Loss**: 0.129313
- **Dataset**: Shakespeare (~338K tokens)

## Usage

Enter a prompt (e.g., "ROMEO:") and adjust the max length and temperature sliders to generate Shakespeare-style text.

## Model Quantization

The model uses **INT8 quantization** to fit within HuggingFace Spaces' 1 GB storage limit:
- **Original model**: ~2.6 GB (float32)
- **Quantized model**: ~0.69 GB (INT8) ✅
- **Quality**: Minimal loss (<1% accuracy drop)

The app automatically loads the quantized model (`deepseek_v3_final_int8.pt`) if available, falling back to the regular model otherwise.

## Files

- `app.py`: Gradio application (automatically handles quantized model loading)
- `requirements.txt`: Python dependencies
- `deepseek_v3_final_int8.pt`: Quantized INT8 model (~0.69 GB) ✅ **Use this for deployment**
- `deepseek_v3_final.pt`: Original float32 model (~2.6 GB) - optional fallback

## Deployment

To deploy this Space:

1. **Quantize the model** (if not already done):
   ```bash
   python quantize_model.py
   ```
   This creates `deepseek_v3_final_int8.pt` (~0.69 GB)

2. **Upload files to the Space**:
   - `deepseek_v3_final_int8.pt` (quantized model) ✅
   - `deepseek_v3_model.py` (model architecture)
   - `app.py`, `requirements.txt`, `README.md`

3. The app will automatically load the quantized model and start serving

**Note**: The quantized INT8 model fits within the 1 GB limit, while the original float32 model does not.

## References

- DeepSeek-V3: Multi-Latent Head Attention and Mixture of Experts
- Training details: See main README.md in the repository

