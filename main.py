# To run this script, you first need to install the 'datasets' library:
# pip install datasets transformers torch

from dataset_loader import load_dataset_wrapper
from pre_processing_dataset import process_dataset
from model import create_model
from trainer import Trainer
import torch

def main():
    """
    Full pipeline: Load data, create a model, and train it.
    """
    # --- Device Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading and Processing ---
    print("\n--- Data Loading and Processing ---")
    try:
        raw_dataset = load_dataset_wrapper()
        processed_dataset = process_dataset(raw_dataset)
        print("Dataset ready.")
    except Exception as e:
        print(f"\nAn error occurred during data processing: {e}")
        return

    # --- Model Creation ---
    print("\n--- Model Creation ---")
    try:
        model = create_model()
        model.to(device) # IMPORTANT: Move the model to the GPU
    except Exception as e:
        print(f"\nAn error occurred during model creation: {e}")
        return

    # --- Training ---
    print("\n--- Training ---")
    try:
        # Create a Trainer instance, which also creates the optimizer
        trainer = Trainer(model, device)
        
        # Start training!
        # We'll start with a small number of steps (500) just to see it work.
        # A real training run would be for hundreds of thousands of steps.
        trainer.train(processed_dataset, num_train_steps=500, log_interval=100)

        # Save the trained model
        trainer.save_model()

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")


if __name__ == "__main__":
    main()