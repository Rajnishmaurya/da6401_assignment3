Hindi Transliteration with Seq2Seq Models
This repository contains a Sequence-to-Sequence (Seq2Seq) model implementation for transliterating English/Latin text to Hindi (Devanagari script). The model is built using PyTorch and integrates with Weights & Biases (wandb) for experiment tracking and hyperparameter optimization.

Project Overview
Transliteration is the process of converting text from one script to another while preserving pronunciation. In this project, we convert English/Latin characters to Hindi Devanagari script, a common task for Indian language processing applications.

Model Architecture
Our implementation uses a Seq2Seq architecture with:

Encoder-decoder structure
Choice of RNN, LSTM, or GRU cells
Embedding layers for both source and target languages
Configurable hidden dimensions and number of layers

Optional beam search decoding
python
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, emb_dim=128, hidden_size=256, 
                 rnn_type='LSTM', num_layers=1, device='cpu'):
        super(Seq2Seq, self).__init__()
        # Model architecture definition
        # ...
Hyperparameter Optimization
We used Weights & Biases sweeps to systematically search for optimal hyperparameters. Our sweep configuration:

Parameter	Type	Values/Range
embed_size	Discrete	[32, 64, 128]
hidden_size	Discrete	[32, 64, 128]
num_layers	Discrete	[1, 2]
cell_type	Discrete	["RNN", "GRU", "LSTM"]
dropout	Discrete	[0.2, 0.3]
lr	Continuous	Min: 0.0001, Max: 0.01
batch_size	Discrete	[32, 64]
beam_size	Discrete	[1, 3, 5]
Key Findings
Based on our hyperparameter optimization experiments, we discovered:

Cell Type Performance:
LSTM and GRU models consistently outperform RNN models
LSTM achieved the highest accuracy (~0.70-0.75)

Architecture Dimensions:
Higher values for hidden_size (100-130) and embed_size (100-130) yielded better results
2-layer models outperformed single-layer models

Regularization:
Moderate dropout values (0.28-0.29) were crucial for best performance
Very low dropout (0.20-0.22) resulted in poorer performance, suggesting overfitting

Training Parameters:
Optimal learning rates were in the mid-range (0.002-0.004)
Larger batch_size values (60-65) performed better

Installation

git clone https://github.com/yourusername/hindi-transliteration.git

cd hindi-transliteration
pip install -r requirements.txt
Training
bash
python train.py --embed_size 128 --hidden_size 128 --cell_type LSTM --num_layers 2 --dropout 0.29
Running a Sweep
bash
python sweep.py
Inference
bash
python predict.py --input "namaste" --model_path "models/best_model.pth"
Future Work
Architecture Improvements:
Implement attention mechanisms
Explore Transformer-based architectures
Add bidirectional encoder support
Error-Focused Training:
Create specialized datasets for challenging characters
Implement targeted error correction mechanisms
Enhanced Evaluation:
Develop more nuanced evaluation metrics beyond accuracy
Create specific test sets for different error categories
Requirements
Python 3.7+
PyTorch 1.7+
wandb
numpy
pandas
tqdm
