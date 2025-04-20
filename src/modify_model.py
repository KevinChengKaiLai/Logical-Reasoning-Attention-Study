import torch
import torch.nn as nn
from transformers import RobertaForMultipleChoice, AutoConfig
from load_PTmodel import load_roberta_for_multiple_choice # Assuming load_PTmodel.py is in the same directory
import copy
from argparse import ArgumentParser

def modify_roberta_attention(model: RobertaForMultipleChoice, depth: int, width: int) -> RobertaForMultipleChoice:
    """
    Modifies a pre-trained RoBERTa model for multiple choice by adjusting
    the depth and width (attention heads) of its encoder layers.

    Args:
        model (RobertaForMultipleChoice): The pre-trained model instance.
        depth (int): The number of initial encoder layers to keep (1, 2, or 3).
        width (int): The number of attention heads for the kept layers (4, 8, or 12).

    Returns:
        RobertaForMultipleChoice: A new model instance with modified configuration.

    Raises:
        ValueError: If hidden size is not divisible by the desired width,
                    or if depth/width values are invalid.
        AttributeError: If the model structure is unexpected.
    """
    if depth not in [1, 2, 3]:
        raise ValueError(f"Invalid depth: {depth}. Must be 1, 2, or 3.")
    if width not in [4, 8, 12]:
        raise ValueError(f"Invalid width: {width}. Must be 4, 8, or 12.")

    # --- Create a deep copy to avoid modifying the original model ---
    print(f"Creating deep copy of the model for modification (Depth={depth}, Width={width})...")
    modified_model = copy.deepcopy(model)
    config = modified_model.config
    hidden_size = config.hidden_size

    # --- Validate configuration compatibility ---
    if hidden_size % width != 0:
        raise ValueError(f"Hidden size ({hidden_size}) must be divisible by the number of attention heads ({width}).")

    new_attention_head_size = hidden_size // width
    print(f"Modifying first {depth} layers to have {width} attention heads (head size: {new_attention_head_size}).")
    print("WARNING: This will re-initialize Query, Key, Value, and Attention Output weights for modified layers.")

    # --- Modify the specified number of layers ---
    original_layers = modified_model.roberta.encoder.layer
    if depth > len(original_layers):
        raise ValueError(f"Requested depth ({depth}) exceeds model's layer count ({len(original_layers)}).")

    modified_layers_list = []
    for i in range(depth):
        layer = original_layers[i]
        try:
            # Access the self-attention module
            self_attention = layer.attention.self

            # Update head configuration
            self_attention.num_attention_heads = width
            self_attention.attention_head_size = new_attention_head_size
            self_attention.all_head_size = hidden_size # Should be hidden_size (width * head_size)

            # --- Re-initialize weight matrices for Q, K, V, and Attention Output ---
            # These layers' dimensions depend on the head configuration indirectly or directly
            # Re-initializing is necessary when head config changes but hidden_size stays the same.
            self_attention.query = nn.Linear(hidden_size, hidden_size)
            self_attention.key = nn.Linear(hidden_size, hidden_size)
            self_attention.value = nn.Linear(hidden_size, hidden_size)
            # The attention output dense layer combines heads, re-initialize too
            layer.attention.output.dense = nn.Linear(hidden_size, hidden_size)

            # Re-initialize weights (using default initialization)
            self_attention.query.reset_parameters()
            self_attention.key.reset_parameters()
            self_attention.value.reset_parameters()
            layer.attention.output.dense.reset_parameters()

            modified_layers_list.append(layer)

        except AttributeError as e:
            raise AttributeError(f"Could not access expected attributes in layer {i}. Model structure might differ. Error: {e}")

    # --- Replace the encoder's layer list with the modified subset ---
    modified_model.roberta.encoder.layer = nn.ModuleList(modified_layers_list)

    # --- Update config (optional, mainly for bookkeeping) ---
    # Note: This config change reflects the modified part, not necessarily the original base model config fully
    modified_model.config.num_hidden_layers = depth
    # modified_model.config.num_attention_heads = width # Be careful modifying this if layers could differ

    print(f"Model modification complete. Using first {depth} layers, each with {width} heads.")
    print("WARNING: Total parameter count is significantly reduced due to using fewer layers.")

    return modified_model

# --- Example Usage ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Modify RoBERTa attention layers.")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Base RoBERTa model name.')
    parser.add_argument('--depth', type=int, required=True, help='Number of layers to keep (1, 2, or 3).')
    parser.add_argument('--width', type=int, required=True, help='Number of attention heads (4, 8, or 12).')

    args = parser.parse_args()

    # 1. Load the base model
    base_model = load_roberta_for_multiple_choice(args.model_name)

    if base_model:
        # 2. Modify the model
        try:
            modified_model = modify_roberta_attention(base_model, args.depth, args.width)

            # 3. Verify (optional)
            print("\n--- Verification ---")
            print(f"Number of layers in modified model's encoder: {len(modified_model.roberta.encoder.layer)}")
            if len(modified_model.roberta.encoder.layer) > 0:
                first_layer_config = modified_model.roberta.encoder.layer[0].attention.self
                print(f"Heads in first layer: {first_layer_config.num_attention_heads}")
                print(f"Head size in first layer: {first_layer_config.attention_head_size}")

            # Count parameters (example)
            original_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            modified_params = sum(p.numel() for p in modified_model.parameters() if p.requires_grad)
            print(f"\nParameter Count:")
            print(f"Original model: {original_params:,}")
            print(f"Modified model: {modified_params:,}")


        except (ValueError, AttributeError) as e:
            print(f"\nError during modification: {e}")