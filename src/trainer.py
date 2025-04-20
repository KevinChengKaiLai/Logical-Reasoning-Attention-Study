# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # Optional: for progress bars
import os
import copy
import matplotlib.pyplot as plt

def plot_training_metrics(train_loss_lst, val_loss_lst, train_accuracy_lst, val_accuracy_lst, model_depth,model_width, epochs):
    """
    Plots training metrics and saves the figures to specified directories.
    
    Args:
        train_loss_lst (list): Training loss values for each epoch
        val_loss_lst (list): Validation loss values for each epoch
        train_accuracy_lst (list): Training accuracy values for each epoch
        val_accuracy_lst (list): Validation accuracy values for each epoch
        model_depth (int): The depth (number of layers used) in the model
        epochs (int): Total number of epochs trained
    """
    # Create directory if it doesn't exist
    os.makedirs("metric", exist_ok=True)
    
    # Create x-axis for epochs (starting from 1)
    epoch_nums = list(range(1, len(train_loss_lst) + 1))
    
    # Create figure for loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_nums, train_loss_lst, 'b-', label='Training Loss')
    plt.plot(epoch_nums, val_loss_lst, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Depth={model_depth},Width={model_width})')
    plt.legend()
    plt.grid(True)
    
    # Save loss figure
    loss_filename = f"metric/loss_depth{model_depth}_width_{model_width}_epochs{epochs}.png"
    plt.savefig(loss_filename)
    print(f"Loss plot saved to {loss_filename}")
    plt.close()
    
    # Create figure for accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_nums, train_accuracy_lst, 'b-', label='Training Accuracy')
    plt.plot(epoch_nums, val_accuracy_lst, 'r-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy (Depth={model_depth},Width={model_width})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy is between 0 and 1
    
    # Save accuracy figure
    accuracy_filename = f"metric/accuracy_depth{model_depth}_width_{model_width}_epochs{epochs}.png"
    plt.savefig(accuracy_filename)
    print(f"Accuracy plot saved to {accuracy_filename}")
    plt.close()


def train_model(model, train_dataloader, val_dataloader, learning_rate, epochs, device, save_path, model_depth,model_width):
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
    
    # make list for metric plotting
    train_loss_lst = []
    val_loss_lst = []
    train_accuracy_lst = []
    val_accuracy_lst = []
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Training phase
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_predictions = 0
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

            logits = outputs.logits # Shape: [batch_size, num_choices]
            predictions = torch.argmax(logits, dim=-1) # Get predicted choice index
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar description (optional)
            train_iterator.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions
        
        train_loss_lst.append(avg_train_loss)
        train_accuracy_lst.append(train_accuracy)
        
        print(f"Average Training Loss: {avg_train_loss:.3f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")

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
        
        val_loss_lst.append(avg_val_loss)
        val_accuracy_lst.append(val_accuracy)
        
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
        
    # Plot training metrics after training is complete
    print("\n--- Plotting Training Metrics ---")
    plot_training_metrics(
        train_loss_lst, 
        val_loss_lst, 
        train_accuracy_lst, 
        val_accuracy_lst, 
        model_depth, 
        model_width,
        epochs
    )
    print("\n--- Finished Plotting Training Metrics ---")


