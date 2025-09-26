import torch
from model import create_model, GPTConfig
from transformers import AutoTokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, device='cpu'):
    """
    Generates text using the model.
    """
    print(f"Generating text from prompt: '{prompt}'")
    
    # Put the model in evaluation mode
    model.eval()
    
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate tokens one by one
    generated_ids = input_ids
    
    with torch.no_grad(): # We don't need to calculate gradients for generation
        for _ in range(max_new_tokens):
            # Get the logits from the model
            # We only pass the last `n_positions` tokens to the model to avoid size errors
            # This is our "context window"
            context = generated_ids[:, -GPTConfig().n_positions:]
            logits = model(context)
            
            # The model predicts the next token, so we only care about the logits for the very last token
            last_logits = logits[:, -1, :]
            
            # Use the logits to sample the next token ID
            # This adds some randomness, making the output more interesting than just picking the single best token
            next_token_id = torch.multinomial(torch.nn.functional.softmax(last_logits, dim=-1), num_samples=1)
            
            # Append the new token to our sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    # Decode the generated token IDs back into text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------\n")


if __name__ == "__main__":
    # --- Device Setup ---
    # Check if a CUDA-enabled GPU is available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Model Loading ---
    checkpoint_path = "../model.pt"
    print(f"Loading model from {checkpoint_path}...")
    
    # Create a new model instance first
    model = create_model()
    
    # Load the saved state dictionary
    # The state_dict contains all the learned weights and biases.
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Move the model to the selected device
    model.to(device)
    print("Model loaded successfully.")
    
    # 2. Load the tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 3. Define a prompt
    prompt_text = "The meaning of life is"
    
    # 4. Generate text, passing the device to fix the error
    generate_text(model, tokenizer, prompt_text, device=device)
    
    prompt_text_2 = "Once upon a time in a land far, far away"
    generate_text(model, tokenizer, prompt_text_2, device=device)
