# To run this script, you first need to install the 'datasets' library:
# pip install datasets transformers torch

from dataset_loader import load_dataset_wrapper
from pre_processing_dataset import process_dataset
from model import create_model
from trainer import Trainer
import torch
import os

def main():
    """
    Full pipeline: Load data, create a model, and train it.
    """
    # --- Configuration ---
    num_train_steps = 50000
    log_interval = 100
    save_interval = 5000 # Save a checkpoint every 5000 steps
    resume_from_checkpoint = "../model_final.pt" # Set to your last checkpoint to resume

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
        model.to(device)
        
        # --- RESUME FROM CHECKPOINT ---
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            model.load_state_dict(torch.load(resume_from_checkpoint))
        else:
            print("Starting training from scratch.")

    except Exception as e:
        print(f"\nAn error occurred during model creation: {e}")
        return

    # --- Training ---
    print("\n--- Training ---")
    try:
        trainer = Trainer(model, device)
        
        trainer.train(
            processed_dataset, 
            num_train_steps=num_train_steps, 
            log_interval=log_interval, 
            save_interval=save_interval
        )

        trainer.save_model(checkpoint_path=f"../model_final_{num_train_steps}.pt")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")


if __name__ == "__main__":
    main()