# ReClor RoBERTa Attention Study

A systematic study of attention mechanisms in RoBERTa models for logical reasoning tasks using the ReClor dataset. This project investigates how different attention configurations (depth and width) affect performance on multiple-choice logical reasoning questions.

## Project Overview

This research project explores the relationship between attention mechanism configurations and logical reasoning performance by:

- **Modifying RoBERTa architectures** with different layer depths (1, 2, 3) and attention head widths (4, 8, 12)
- **Fine-tuning only attention components** while keeping other parameters frozen
- **Evaluating performance** on the ReClor (Reading Comprehension with Logical Reasoning) dataset
- **Analyzing training dynamics** through comprehensive metrics and visualizations

## Dataset

The project uses the **ReClor dataset**, which contains:
- Logical reasoning questions in multiple-choice format
- Context passages requiring analytical reasoning
- 4 answer choices per question with single correct answer
- Training and validation splits

## Architecture

### Base Model
- **RoBERTa** (Robustly Optimized BERT Pretraining Approach)
- Multiple Choice head for 4-way classification
- Transformer encoder architecture

### Modifications
- **Depth variations**: 1, 2, or 3 encoder layers
- **Width variations**: 4, 8, or 12 attention heads per layer
- **Selective training**: Only attention weights (Q, K, V, Output) are trainable
- **Parameter efficiency**: Significant reduction in trainable parameters

## Quick Start

### Prerequisites
```bash
pip install torch transformers tqdm matplotlib
```

### Basic Usage

1. **Prepare your data**: Ensure ReClor dataset files are in JSON format
2. **Run a training experiment**:
```bash
python src/run.py \
    --train_data_path path/to/train.json \
    --val_data_path path/to/val.json \
    --depth 2 \
    --width 8 \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 1e-5
```

### Example Commands

```bash
# Train with 1 layer, 4 attention heads
python src/run.py --train_data_path data/train.json --val_data_path data/val.json --depth 1 --width 4 --epochs 5

# Train with 3 layers, 12 attention heads
python src/run.py --train_data_path data/train.json --val_data_path data/val.json --depth 3 --width 12 --epochs 3 --batch_size 16
```

## Project Structure

```
ReClor-RoBERTa-Attention-Study/
├── src/
│   ├── dataset.py          # ReClor dataset loading and preprocessing
│   ├── load_PTmodel.py     # Pre-trained RoBERTa model loading
│   ├── modify_model.py     # Attention architecture modifications
│   ├── trainer.py          # Training loop and evaluation
│   └── run.py              # Main execution script
├── saved_models/           # Trained model checkpoints
├── metric/                 # Training metrics and plots
├── .gitignore
└── README.md
```

## Core Components

### 1. Dataset Processing (dataset.py)
- **ReclorDataset**: PyTorch Dataset class for ReClor data
- **Tokenization**: RoBERTa tokenizer with sequence pairing
- **Batch Processing**: Efficient DataLoader creation

### 2. Model Loading (load_PTmodel.py)
- **Pre-trained Models**: Loads RoBERTa for multiple choice
- **Configuration Management**: Handles model configurations
- **Error Handling**: Robust model loading with validation

### 3. Architecture Modification (modify_model.py)
- **Attention Reconfiguration**: Modifies depth and width
- **Weight Reinitialization**: Resets attention parameters
- **Parameter Validation**: Ensures configuration compatibility

### 4. Training System (trainer.py)
- **Selective Training**: Only attention weights are trainable
- **Performance Tracking**: Loss and accuracy monitoring
- **Model Checkpointing**: Saves best performing models
- **Visualization**: Automatic metric plotting

### 5. Experiment Runner (run.py)
- **CLI Interface**: Command-line argument parsing
- **Pipeline Integration**: Coordinates all components
- **Configuration Management**: Handles hyperparameters

## Training Features

### Selective Parameter Training
- **Frozen Parameters**: All non-attention weights remain fixed
- **Trainable Components**: Only Q, K, V, and attention output layers
- **Efficiency**: Dramatically reduced parameter count

### Comprehensive Monitoring
- **Training Metrics**: Loss and accuracy per epoch
- **Validation Tracking**: Performance on held-out data
- **Best Model Saving**: Automatic checkpoint of best performance
- **Visual Analytics**: Automatic plot generation

### Output Artifacts
- **Model Checkpoints**: `saved_models/roberta_depth{D}_width{W}_epoch{E}_best.pt`
- **Training Plots**: `metric/loss_depth{D}_width_{W}_epochs{E}.png`
- **Accuracy Plots**: `metric/accuracy_depth{D}_width_{W}_epochs{E}.png`

## Command Line Arguments

### Model Configuration
- `--model_name`: Base RoBERTa model (`roberta-base`, `roberta-large`)
- `--depth`: Number of encoder layers (1, 2, 3)
- `--width`: Number of attention heads (4, 8, 12)

### Data Parameters
- `--train_data_path`: Path to training JSON file
- `--val_data_path`: Path to validation JSON file
- `--max_length`: Maximum sequence length (default: 512)

### Training Hyperparameters
- `--learning_rate`: AdamW learning rate (default: 1e-5)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training/validation (default: 8)

### Output Settings
- `--output_dir`: Directory for saved models (default: `saved_models`)

## Expected Results

### Model Variants
The project systematically evaluates 9 configurations:
- **3 depth options** × **3 width options** = 9 total experiments
- Each configuration produces unique performance characteristics
- Attention patterns vary significantly across configurations

### Performance Metrics
- **Accuracy**: Multiple-choice classification accuracy
- **Loss**: Cross-entropy loss during training
- **Convergence**: Training dynamics and stability
- **Efficiency**: Parameter count and training time

## Research Applications

### Attention Mechanism Analysis
- **Depth vs. Performance**: How layer count affects reasoning
- **Width vs. Accuracy**: Impact of attention head count
- **Parameter Efficiency**: Optimal configurations for given constraints

### Logical Reasoning Studies
- **Reasoning Patterns**: What attention learns for logical tasks
- **Generalization**: How attention transfers across question types
- **Interpretability**: Understanding model decision processes

## Important Notes

### Hardware Requirements
- **GPU Recommended**: CUDA-compatible device preferred
- **Memory**: Sufficient RAM for batch processing
- **Storage**: Space for model checkpoints and metrics

### Training Considerations
- **Initialization**: Attention weights are randomly reinitialized
- **Convergence**: May require hyperparameter tuning
- **Validation**: Always use separate validation set

### Model Limitations
- **Reduced Capacity**: Fewer layers = lower model capacity
- **Task Specific**: Optimized for multiple-choice reasoning
- **Architecture Dependent**: Results may vary with different base models

## Future Work

- **Extended Architectures**: Test with other transformer models
- **Attention Visualization**: Add attention heatmap generation
- **Hyperparameter Search**: Automated optimization
- **Cross-Dataset Evaluation**: Test on other reasoning datasets
- **Attention Analysis**: Deeper investigation of learned patterns

## References

- **RoBERTa**: Liu, Y., et al. "RoBERTa: A robustly optimized BERT pretraining approach." arXiv preprint arXiv:1907.11692 (2019).
- **ReClor Dataset**: Yu, W., et al. "ReClor: A reading comprehension dataset requiring logical reasoning." arXiv preprint arXiv:2002.04326 (2020).
- **Attention Mechanisms**: Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

---

For questions or issues, please open a GitHub issue or contact the maintainers.
