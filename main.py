# To run this script, you first need to install the 'datasets' library:
# pip install datasets transformers torch

from dataset_loader import load_dataset_wrapper
from pre_processing_dataset import process_dataset

def main():
    """
    Downloads, processes, and investigates the OpenWebText dataset.
    """
    print("Attempting to load the OpenWebText dataset in streaming mode...")

    try:
        # Step 1: Load the raw text dataset
        raw_dataset = load_dataset_wrapper()
        print("Raw dataset loaded successfully in streaming mode.")

        # Step 2: Tokenize and process the dataset
        processed_dataset = process_dataset(raw_dataset)

        # Let's look at the first example from the PROCESSED dataset
        print("\nInvestigating the first processed example from the dataset:")
        first_processed_example = next(iter(processed_dataset))

        # Print the keys to see what's in our processed example
        print(f"\nKeys in the processed example: {list(first_processed_example.keys())}")

        # Print the shape/size of the input_ids
        print(f"Number of tokens in the first example's 'input_ids': {len(first_processed_example['input_ids'])}")
        print(f"This should match our CONTEXT_LENGTH of 1024.")

        # You can optionally print the first 20 token IDs to see what they look like
        print(f"\nFirst 20 token IDs: {first_processed_example['input_ids'][:20]}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have an internet connection and the required libraries are installed correctly.")
        print("You can install them with: pip install datasets transformers torch")

if __name__ == "__main__":
    main()