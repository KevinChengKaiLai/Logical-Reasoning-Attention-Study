# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # Optional: for progress bars
import os
import copy

def train_model(model, train_dataloader, val_dataloader, learning_rate, epochs, device, save_path, model_depth):
    """
    Trains and evaluates a modified RoBERTa model for the ReClor task.

    Args:
        model (nn.Module): The modified RobertaForMultipleChoice model instance.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the best model state dictionary.
        model_depth (int): The depth (number of layers used) in the modified model,
                           needed for correct parameter freezing.
    """
    print("--- Starting Training ---")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Model Save Path: {save_path}")

    model.to(device)

    # --- Parameter Freezing ---
    print("Freezing parameters...")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Freeze everything by default
        param.requires_grad = False

    # Unfreeze only the specified attention components in the active layers
    print(f"Unfreezing self-attention components in the first {model_depth} layers...")
    for i in range(model_depth):
        try:
            layer = model.roberta.encoder.layer[i]
            # Unfreeze Q, K, V in self-attention
            for name, param in layer.attention.self.named_parameters():
                 if name in ['query.weight', 'query.bias', 'key.weight', 'key.bias', 'value.weight', 'value.bias']:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    # print(f"  Unfrozen: layer {i} attention.self.{name}")

            # Unfreeze the output projection of self-attention
            for name, param in layer.attention.output.named_parameters():
                 if name == 'dense.weight' or name == 'dense.bias':
                    param.requires_grad = True
                    trainable_params += param.numel()
                    # print(f"  Unfrozen: layer {i} attention.output.{name}")

        except IndexError:
            print(f"Warning: Tried to access layer {i}, but model depth is only {model_depth}.")
            break
        except AttributeError as e:
            print(f"Warning: Could not access expected structure in layer {i}. Skipping unfreezing for this layer. Error: {e}")

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (Attention Q/K/V/Output in first {model_depth} layers): {trainable_params:,}")

    # Filter parameters for the optimizer
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # --- Training Loop ---
    best_val_accuracy = -1.0
    best_model_state = None

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Training phase
        model.train()
        total_train_loss = 0
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training") # Optional progress bar

        for batch in train_iterator:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # RoBERTa for Multiple Choice expects input_ids and attention_mask
            # to be shape [batch_size, num_choices, sequence_length]
            # and labels to be shape [batch_size]
            batch_size, num_choices, seq_len = input_ids.shape

            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar description (optional)
            train_iterator.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation") # Optional progress bar

        with torch.no_grad():
            for batch in val_iterator:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                batch_size, num_choices, seq_len = input_ids.shape

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels # Include labels to get loss, but we'll use logits for accuracy
                )

                loss = outputs.loss
                logits = outputs.logits # Shape: [batch_size, num_choices]

                total_val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1) # Get predicted choice index
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Use deepcopy to ensure the state isn't affected by further training
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"*** New best model saved with validation accuracy: {best_val_accuracy:.4f} ***")

    # --- Save the best model state after all epochs ---
    if best_model_state:
        print(f"\n--- Training Finished ---")
        print(f"Best Validation Accuracy achieved: {best_val_accuracy:.4f}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"Best model state saved to {save_path}")
    else:
        print("\n--- Training Finished ---")
        print("No best model state was saved (validation may not have improved).")

# --- Example Skeleton for how to use the trainer ---
if __name__ == '__main__':
    # This block is for demonstration; you'd typically call train_model
    # from your main experiment script.

    # 1. Define Hyperparameters (replace with your actual values)
    LEARNING_RATE = 1e-5 # Example value
    EPOCHS = 3          # Example value
    BATCH_SIZE = 8      # Example value
    MODEL_NAME = 'roberta-base'
    # --- Specify the configuration you are training ---
    DEPTH = 1           # Example value (must match the model being loaded/modified)
    WIDTH = 12          # Example value (must match the model being loaded/modified)
    # --- ---
    SAVE_DIR = "saved_models"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"roberta_depth{DEPTH}_width{WIDTH}_best.pt")

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load DataLoaders (replace with your actual loading logic)
    # You need to use dataset.py and load_data function here
    # Example placeholder:
    print("Placeholder: Load your train_dataloader and val_dataloader here.")
    # from dataset import ReclorDataset, load_data
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # train_dataset = ReclorDataset(data_path='path/to/train.json', tokenizer=tokenizer)
    # val_dataset = ReclorDataset(data_path='path/to/val.json', tokenizer=tokenizer)
    # train_loader = load_data(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = load_data(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_loader = None # Replace with actual loader
    val_loader = None   # Replace with actual loader

    # 4. Load and Modify Model (replace with your actual loading logic)
    print("Placeholder: Load base model and modify it here.")
    # from load_PTmodel import load_roberta_for_multiple_choice
    # from modify_model import modify_roberta_attention
    # base_model = load_roberta_for_multiple_choice(MODEL_NAME)
    # modified_model = modify_roberta_attention(base_model, depth=DEPTH, width=WIDTH)
    modified_model = None # Replace with actual model

    # 5. Run Training
    if modified_model and train_loader and val_loader:
         print("\nStarting trainer function...")
         # train_model(
         #     model=modified_model,
         #     train_dataloader=train_loader,
         #     val_dataloader=val_loader,
         #     learning_rate=LEARNING_RATE,
         #     epochs=EPOCHS,
         #     device=device,
         #     save_path=MODEL_SAVE_PATH,
         #     model_depth=DEPTH # Pass the depth used for modification
         # )
         print("Example complete. Uncomment and fill placeholders to run.")
    else:
        print("Skipping train_model call because model or dataloaders are not defined in example.")