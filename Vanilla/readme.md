# Transliteration Model using Sequence-to-Sequence Architecture

This repository contains an implementation of a Sequence-to-Sequence (Seq2Seq) model for transliteration, specifically for Hindi-English (Romanization) transliteration using the Dakshina dataset.

## Project Overview

The model transliterates text from one script to another. In this implementation, it focuses on Hindi to Latin script (Romanization) conversion. The architecture uses encoder-decoder recurrent neural networks with various cell types (RNN, GRU, LSTM) and supports beam search decoding.

## Features

- Sequence-to-Sequence model with configurable architecture:
  - Multiple RNN cell types (RNN, GRU, LSTM)
  - Customizable embedding dimensions
  - Variable hidden layer sizes
  - Support for multiple layers
  - Dropout regularization
- Beam search decoding for improved transliteration quality
- Flexible dataset handling and preprocessing
- Hyperparameter optimization through Weights & Biases
- Comprehensive evaluation metrics

## Requirements

- Python 3.6+
- PyTorch 1.0+
- pandas
- wandb (Weights & Biases)
- tqdm

## Installation

```bash
pip install torch pandas wandb tqdm
```

## Data

This model uses the Dakshina dataset for Hindi-English transliteration. The dataset should be organized as follows:

```
/kaggle/input/dakshina-dataset/hi/lexicons/
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

Each file contains tab-separated values with the following format:
```
target_word    source_word    count
```

## Usage

### Training

To train the model with default parameters:

```python
python vanilla.py
```

### Hyperparameter Optimization

The code includes a hyperparameter sweep configuration using Weights & Biases:

```python
sweep_id = wandb.sweep(sweep_config, project="dakshina-transliteration")
wandb.agent(sweep_id, function=main, count=25)
```

### Evaluation

The model can be evaluated and predictions saved to a CSV file:

```python
evaluate_and_save(model, test_loader, input_vocab, output_vocab, device, csv_path="test_predictions.csv")
```

## Model Architecture

### Encoder

The encoder consists of:
- Embedding layer: Converts input tokens to dense vectors
- RNN layer: Processes the sequence and produces a hidden state

### Decoder

The decoder consists of:
- Embedding layer: Converts output tokens to dense vectors
- RNN layer: Generates output sequence based on encoder hidden state
- Linear layer: Projects RNN outputs to vocabulary space

### Training Process

1. The encoder processes the source sequence and produces a hidden state
2. The decoder uses this hidden state to generate the target sequence
3. Teacher forcing is used to stabilize training
4. Cross-entropy loss is minimized using Adam optimizer

### Inference

During inference, two decoding strategies are available:
- Greedy decoding: Select the most probable token at each step
- Beam search: Maintain multiple hypotheses and select the most probable sequence

## Results

The best model architecture found during experimentation:
- Embedding size: 128
- Hidden size: 128
- Number of layers: 2
- Cell type: LSTM
- Dropout: 0.3
- Learning rate: ~0.0019
- Batch size: 64

## Link
[Github Link](https://github.com/Rajnishmaurya/da6401_assignment3/tree/main/Vanilla)  
[Wandb Report](https://api.wandb.ai/links/da24m015-iitm/xhh9mouq)