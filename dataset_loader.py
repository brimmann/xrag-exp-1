from datasets import load_dataset


def load_dataset_wrapper():
    """
    Downloads and investigates the OpenWebText dataset using streaming.
    """
    dataset = load_dataset("sytelus/openwebtext", streaming=True, split="train")
    return dataset
