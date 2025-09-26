import torch
from model import create_model, GPTConfig
from transformers import AutoTokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cpu'):
    """
    Generates text using the model.
    """
    print(f"Generating text from prompt: '{prompt}'")
    
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids
    
    # --- DEBUGGING ---
    # Let's check the device of our model and our initial data.
    model_device = next(model.parameters()).device
    print(f"  [Debug] Model is on device: {model_device}")
    print(f"  [Debug] Initial 'generated_ids' is on device: {generated_ids.device}")
    # --- END DEBUGGING ---

    with torch.no_grad():
        for i in range(max_new_tokens):
            context = generated_ids[:, -GPTConfig().n_positions:]
            
            # --- DEBUGGING ---
            # Now let's check the context right before it's sent to the model on each loop.
            print(f"  [Debug] Loop {i+1}: 'context' tensor is on device: {context.device}")
            if context.device != model_device:
                print("  [Debug] MISMATCH DETECTED! Context is not on the same device as the model.")
            # --- END DEBUGGING ---

            logits = model(context)
            
            last_logits = logits[:, -1, :]
            next_token_id = torch.multinomial(torch.nn.functional.softmax(last_logits, dim=-1), num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------\n")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    checkpoint_path = "../model.pt"
    print(f"Loading model from {checkpoint_path}...")
    
    model = create_model()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    print("Model loaded successfully.")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    prompt_text = "The meaning of life is"
    generate_text(model, tokenizer, prompt_text, device=device)
    
    prompt_text_2 = "Once upon a time in a land far, far away"
    generate_text(model, tokenizer, prompt_text_2, device=device)
