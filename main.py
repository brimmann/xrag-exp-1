# To run this script, you first need to install the 'datasets' library:
# pip install datasets

from datasets import load_dataset

def main():
    """
    Downloads and investigates the OpenWebText dataset using streaming.
    """
    print("Attempting to load the OpenWebText dataset in streaming mode...")

    try:
        # Load the updated OpenWebText dataset from Hugging Face in streaming mode.
        # Using "sytelus/openwebtext", a script-less Parquet version of the dataset.
        dataset = load_dataset("sytelus/openwebtext", streaming=True, split="train")
        print("Dataset loaded successfully in streaming mode.")

        # Let's look at the first example in the dataset
        print("\nInvestigating the first example from the dataset:")

        # In streaming mode, we can't index directly, so we get an iterator
        # and grab the first item.
        first_example = next(iter(dataset))

        print(first_example)

        # The example is a dictionary. Let's look at the 'text' content.
        print("\nText from the first example:")
        print(first_example['text'])

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is installed correctly.")
        print("You can install it with: pip install datasets")

if __name__ == "__main__":
    main()