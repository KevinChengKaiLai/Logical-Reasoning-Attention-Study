# run.py

import torch
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer

# Import necessary functions/classes from your other project files
from dataset import ReclorDataset, load_data # Assuming dataset.py is in the same directory or accessible via PYTHONPATH
from load_PTmodel import load_roberta_for_multiple_choice # Assuming load_PTmodel.py is ...
from modify_model import modify_roberta_attention # Assuming modify_model.py is ...
from trainer import train_model # Assuming trainer.py is ...

def main():
    parser = ArgumentParser(description="Run ReClor training experiment for a specific RoBERTa configuration.")

    # --- Model & Configuration Arguments ---
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Base RoBERTa model name.')
    parser.add_argument('--depth', type=int, required=True, help='Number of encoder layers to keep (1, 2, or 3).')
    parser.add_argument('--width', type=int, required=True, help='Number of attention heads (4, 8, or 12).')

    # --- Data Arguments ---
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the ReClor training JSON file.')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to the ReClor validation JSON file.')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for tokenizer.')

    # --- Training Hyperparameters ---
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for AdamW optimizer.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation.')

    # --- Output Arguments ---
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Directory to save the best model.')

    args = parser.parse_args()

    # --- Setup ---
    model_config_name = f"depth{args.depth}_width{args.width}"
    model_save_path = os.path.join(args.output_dir, f"roberta_{model_config_name}_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Experiment Configuration ---")
    print(f"Model Base: {args.model_name}")
    print(f"Target Depth: {args.depth}")
    print(f"Target Width: {args.width}")
    print(f"Max Sequence Length: {args.max_length}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Best Model Save Path: {model_save_path}")
    print("-------------------------------")


    # --- Load Tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- Load DataLoaders ---
    print("\nLoading datasets...")
    try:
        train_dataset = ReclorDataset(data_path=args.train_data_path, tokenizer=tokenizer, max_length=args.max_length)
        val_dataset = ReclorDataset(data_path=args.val_data_path, tokenizer=tokenizer, max_length=args.max_length)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # Use the load_data function from dataset.py
        train_loader = load_data(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = load_data(val_dataset, batch_size=args.batch_size, shuffle=False) # No shuffle for validation
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}. Please check --train_data_path and --val_data_path.")
        return # Exit if data can't be loaded
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- Load and Modify Model ---
    print("\nLoading base model...")
    base_model = load_roberta_for_multiple_choice(args.model_name)
    if not base_model:
        print("Failed to load base model. Exiting.")
        return

    print("\nModifying model...")
    try:
        modified_model = modify_roberta_attention(base_model, depth=args.depth, width=args.width)
    except (ValueError, AttributeError) as e:
        print(f"Error modifying model: {e}. Exiting.")
        return

    # --- Run Training ---
    print("\nStarting training process...")
    try:
        train_model(
            model=modified_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            device=device,
            save_path=model_save_path,
            model_depth=args.depth # Pass the depth used for modification
        )
        print("\n--- run.py finished ---")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == '__main__':
    main()