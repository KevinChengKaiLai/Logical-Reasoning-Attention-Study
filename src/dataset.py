import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from argparse import ArgumentParser
import sys

class ReclorDataset(Dataset):
    """
    PyTorch Dataset class for the ReClor dataset. [cite: 5]
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initializes the ReclorDataset.

        Args:
            data_path (str): Path to the ReClor JSON data file.
            tokenizer: Hugging Face tokenizer instance (e.g., RoBERTa tokenizer).
            max_length (int): Maximum sequence length for tokenization.
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {data_path}")
            sys.exit(1)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Gets a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized inputs, attention mask,
                  label, and the original ID string.
        """
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answers = item['answers'] # List of 4 possible answers
        label = item['label']     # Index of the correct answer (0-3)
        id_string = item['id_string']

        # Prepare inputs for RoBERTa: typically <s> context </s></s> question + answer </s>
        # Here we format it as pairs for batch_encode_plus: [(context, question + answer1), ...]
        inputs = []
        for answer in answers:
            # Combine question and answer, let tokenizer handle separation tokens if needed by model type
            q_and_a = question + ' ' + self.tokenizer.sep_token + ' ' + answer
            inputs.append((context, q_and_a)) # Pass as a pair for sequence pair encoding

        # Tokenize the context-question-answer pairs
        # batch_encode_plus handles the pairing and adds special tokens
        encoding = self.tokenizer.batch_encode_plus(
            inputs,                    # List of pairs [(context, q_a1), (context, q_a2), ...]
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',      # Pad all sequences to max_length
            truncation=True,           # Truncate sequences longer than max_length
            return_tensors='pt'        # Return PyTorch tensors
        )

        # encoding["input_ids"] will be shape [4, max_length]
        # encoding["attention_mask"] will be shape [4, max_length]

        return {
            "input_ids": encoding["input_ids"],           # Shape: [num_answers, max_length]
            "attention_mask": encoding["attention_mask"], # Shape: [num_answers, max_length]
            "label": torch.tensor(label, dtype=torch.long), # Single integer label
            "id_string": id_string                        # Original ID for tracking
        }

def load_data(reclor_data: Dataset, batch_size = 16, shuffle = True):
    """Creates a DataLoader from a ReclorDataset instance."""
    # print(f"Creating DataLoader with batch size: {batch_size}, Shuffle: {shuffle}")

    # Dataset -> DataLoader 
    data_loader = DataLoader(
        reclor_data,
        batch_size=batch_size,
        shuffle=True # Shuffle for training is common
    )

    return data_loader


if __name__ == '__main__':
    parser = ArgumentParser(description="Load ReClor dataset for RoBERTa model.")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the ReClor JSON data file (e.g., train.json, val.json)')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='Name of the pretrained RoBERTa model for tokenizer')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for DataLoader demo')

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    # Initialize the RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading dataset from: {args.data_path}")
    # Create the dataset instance
    reclor_dataset = ReclorDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    print(f"Dataset size: {len(reclor_dataset)}")

    # --- Optional: Demonstrate DataLoader usage ---
    print(f"Creating DataLoader with batch size: {args.batch_size}")
    data_loader = load_data(
        reclor_data=reclor_dataset,
        batch_size=args.batch_size,
        shuffle=True # Explicitly set shuffle, or rely on default
    )


    print("\n--- Example Batch ---")
    # Fetch one batch
    try:
        example_batch = next(iter(data_loader))

        # Print shapes and types of the batch elements
        print("Batch keys:", example_batch.keys())
        print("Input IDs shape:", example_batch["input_ids"].shape)
        print("Attention Mask shape:", example_batch["attention_mask"].shape)
        print("Labels shape:", example_batch["label"].shape)
        print("Labels:", example_batch["label"])
        print("ID Strings:", example_batch["id_string"])

        # Example: Decoding the first sample's first option in the batch
        first_sample_first_option_ids = example_batch["input_ids"][0][0]
        decoded_text = tokenizer.decode(first_sample_first_option_ids, skip_special_tokens=False)
        print("\nDecoded first option of first sample in batch:")
        print(decoded_text)

    except StopIteration:
        print("Dataset is empty or batch size is larger than dataset size.")
    except Exception as e:
        print(f"An error occurred while fetching/processing a batch: {e}")

    print("\nDataset loading script finished.")