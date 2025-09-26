from transformers import AutoTokenizer
from itertools import chain

# Let's define a constant for the context length. 
# This is the number of tokens the model will see at once.
# 1024 is a common size for smaller models like GPT-2.
CONTEXT_LENGTH = 1024

def tokenize_and_group_text(examples, tokenizer):
    """
    This function tokenizes a batch of text examples and then groups them into chunks
    of a fixed size (CONTEXT_LENGTH).
    """
    # Tokenize all the text examples in the batch. The output is a list of lists of token IDs.
    tokenized_outputs = tokenizer(examples["text"])

    # We're not returning the tokenized outputs directly. Instead, we will concatenate all
    # the token lists from the batch and then chunk them into blocks of CONTEXT_LENGTH.
    # This is a common strategy to prepare data for language model pre-training.

    # 1. Concatenate all token lists into one big list.
    concatenated_ids = list(chain.from_iterable(tokenized_outputs["input_ids"]))
    total_length = len(concatenated_ids)

    # 2. We drop the small remainder at the end.
    # You could also pad it, but for pre-training, dropping is simpler.
    total_length = (total_length // CONTEXT_LENGTH) * CONTEXT_LENGTH

    # 3. Split the big list into chunks of CONTEXT_LENGTH.
    result = {
        "input_ids": [concatenated_ids[i : i + CONTEXT_LENGTH] for i in range(0, total_length, CONTEXT_LENGTH)]
    }
    
    # The language model also needs to know which tokens to predict. 
    # For standard language modeling, the labels are the same as the input_ids.
    # The model's job is to predict the next token given the previous ones.
    result["labels"] = result["input_ids"].copy()
    return result


def process_dataset(dataset):
    """
    Applies the tokenization and grouping to the entire dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded successfully.")

    # The .map() function applies our processing function to the dataset.
    # batched=True sends multiple examples to the function at once, which is much faster.
    # remove_columns=['text'] gets rid of the original raw text column after we're done with it.
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_group_text(examples, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    print("Dataset tokenization and processing complete.")
    return tokenized_dataset
