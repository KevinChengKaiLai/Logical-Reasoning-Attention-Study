import torch
from transformers import AutoModelForMultipleChoice, AutoConfig
from argparse import ArgumentParser

def load_roberta_for_multiple_choice(model_name='roberta-base'):
    """
    Loads a pre-trained RoBERTa model with a multiple-choice classification head.

    Args:
        model_name (str): The name of the pre-trained RoBERTa model
                          (e.g., 'roberta-base', 'roberta-large').

    Returns:
        transformers.RobertaForMultipleChoice: The loaded model.
    """
    print(f"Loading pre-trained model: {model_name}")
    try:
        # Load the configuration first to potentially inspect or modify
        config = AutoConfig.from_pretrained(model_name)
        # Load the model configured for multiple choice tasks
        model = AutoModelForMultipleChoice.from_pretrained(model_name, config=config)
        print(f"Model {model_name} loaded successfully.")
        return model
    except OSError as e:
        print(f"Error loading model '{model_name}'. Check the model name and internet connection.")
        print(f"Details: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    parser = ArgumentParser(description="Load a pre-trained RoBERTa model for Multiple Choice.")
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='Name of the pretrained RoBERTa model to load.')

    args = parser.parse_args()

    # Load the model using the function
    model = load_roberta_for_multiple_choice(args.model_name)

    if model:
        print("\n--- Model Configuration Snippet ---")
        # Print some basic config info
        print(model)
        print(f"Model Type: {model.config.model_type}")
        print(f"Number of hidden layers: {model.config.num_hidden_layers}")
        print(f"Number of attention heads: {model.config.num_attention_heads}")
        print(f"Hidden size: {model.config.hidden_size}")

        # You could potentially inspect model layers here too
        # print("\nModel Structure:")
        # print(model)
    else:
        print("Model loading failed.")