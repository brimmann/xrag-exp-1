import torch
import os

class Trainer:
    """
    A class to encapsulate the training loop.
    """
    def __init__(self, model, device, learning_rate=3e-4):
        self.model = model
        self.device = device
        # The AdamW optimizer is a standard and effective choice for training transformers.
        # It takes the model's parameters and a learning rate.
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        print(f"Optimizer created with learning rate: {learning_rate}")

    def save_model(self, checkpoint_path="checkpoints/model.pt"):
        """Saves the model's state dictionary."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        print(f"\n--- Saving model to {checkpoint_path} ---")
        # We save the model's state_dict, which contains all the learned weights and biases.
        torch.save(self.model.state_dict(), checkpoint_path)
        print("Model saved successfully.")

    def train(self, dataset, num_train_steps=500, log_interval=100):
        print("\n--- Starting Training ---")
        self.model.train()  # Set the model to training mode. This enables things like dropout.

        data_iterator = iter(dataset)
        total_loss = 0
        
        for step in range(num_train_steps):
            # 1. GET A BATCH OF DATA
            try:
                batch = next(data_iterator)
            except StopIteration:
                # Our streaming dataset might run out, so we just restart it
                data_iterator = iter(dataset)
                batch = next(data_iterator)

            # Move the data to the GPU and add the batch dimension
            # We use .unsqueeze(0) to change the shape from [1024] to [1, 1024]
            input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to(self.device)
            labels = torch.tensor(batch['labels']).unsqueeze(0).to(self.device)

            # 2. FORWARD PASS: Get the model's predictions (logits)
            logits = self.model(input_ids)
            
            # 3. CALCULATE THE LOSS: How wrong was the model?
            # We compare the model's predicted logits to the actual next tokens (the labels).
            # Cross-entropy loss is the standard for this kind of prediction task.
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 4. BACKWARD PASS: Calculate the gradients
            # This is the magic step. PyTorch automatically calculates how much each of the 
            # 124 million parameters contributed to the error.
            loss.backward()
            
            # 5. UPDATE THE MODEL: Nudge the parameters in the right direction
            # The optimizer uses the gradients to update the model's parameters.
            self.optimizer.step()
            
            # 6. CLEAR THE GRADIENTS for the next step
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            # Periodically log the training progress
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                print(f"Step {step + 1}/{num_train_steps} | Average Loss: {avg_loss:.4f}")
                total_loss = 0 # Reset for the next interval

        print("--- Training Finished ---")
