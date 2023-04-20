# ATCS Practical 1
The first practical of the Advanced Topics in Computational Semantics (ATCS) course for the master AI at the UvA.  
The goal of this practical was threefold:
* to implement four neural models to classify sentence pairs based on their relation;
* to train these models using the Stanford Natural Language Inference (SNLI) corpus [(Bowman
et al., 2015);](https://arxiv.org/pdf/1508.05326.pdf)
* and to evaluate the trained models using the SentEval framework [(Conneau and Kiela, 2018)](https://arxiv.org/pdf/1803.05449.pdf)

## Table of contents
* [General info](#general-info)
* [Prerequisites](#prerequisites)
* [Setup](#setup)
* [Usage](#usage)
* [Acknowledgements](#acknowledgements)


## General info
For the sentence encoder blocks, the following four models are implemented following [Conneau et al. (2018)](https://arxiv.org/pdf/1705.02364.pdf):
* Baseline: averaging word embeddings to obtain sentence representations.
* Unidirectional LSTM, where the last hidden state of the word embeddings is considered as the sentence representation.
* Simple bidirectional LSTM (BiLSTM), where the last hidden state of forward and backward layers are concatenated as sentence representations.
* BiLSTM with max pooling applied to the concatenation of word-level hidden states from both directions to retrieve sentence representations.
	
## Prerequisites
* [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
	
## Setup

1. Create the environment:
```
conda env create -f env.yml
```

2. Install [SentEval](https://github.com/facebookresearch/SentEval) and follow the setup.

## Usage
### Training a model
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`

#### Model Arguments
The models can be trained with the following command line arguments:

```bash
usage: train.py [--model MODEL] [--lr LR] [--lr_decay LR_DECAY]
		    [--lr_decrease_factor LR_DECREASE_FACTOR] [--lr_threshold LR_THRESHOLD] 
		    [--batch_size BATCH_SIZE] [--checkpoint_dir CHECKPOINT_DIR]
		    [--seed SEED] [--log_dir LOG_DIR] [--progress_bar] [--development]

optional arguments:
  --model MODEL                   Name of the encoder model to use. Options: ['aweencoder', 'LSTM', 'BiLSTM', 'BiLSTMMax']. Default is 'aweencoder'.
  --train TRAIN                   Force a training loop to overwrite existing models. Default is set to False.
  --learning_rate LR              Learning rate for the optimizer to use. Default is 0.1.
  --lr_decay LR_DECAY             Learning rate decay applied after each epoch. Default is 0.99.
  --lr_decrease LR_DECREASE       Factor to decrease learning rate with when val accuracy decreases. Default is 5.
  --lr_threshold LR_THRESHOLD     Learning rate threshold to stop training at. Default is 10e-5.
  --batch_size BATCH_SIZE         Minibatch size. Default is 64.
  --num_epochs NUM_EPOCHS         The maximum number of epochs to train for. Default is 20.
  --saving SAVING                 Directory for saving model weights. Default is 'saved'.
  --seed SEED                     Random seed for PyTorch. Default is 17.
  --tensorboard_logs LOG_DIR      The directory for saving logs. Default is 'tensorboard_logs'.
  --debug                         Use 1% of data for debugging purposes.
  ```
  
Example how to train a model from scratch with the command line arguments:

```
$ python train.py --model=LSTM --train=True
```
### Evaluating a model
The models can be evaluated on the SentEval tasks with the following command line arguments:

```bash
usage: eval.py [--model MODEL] [--lr LR] [--lr_decay LR_DECAY]
		    [--lr_decrease_factor LR_DECREASE_FACTOR] [--lr_threshold LR_THRESHOLD] 
		    [--batch_size BATCH_SIZE] [--checkpoint_dir CHECKPOINT_DIR]
		    [--seed SEED] [--log_dir LOG_DIR] [--progress_bar] [--development]

optional arguments:
  --model MODEL                        Name of the encoder model to use. Options: ['aweencoder', 'LSTM', 'BiLSTM', 'BiLSTMMax']. Default is 'aweencoder'.
  --batch_size BATCH_SIZE              Minibatch size. Default is 64.
  --saving SAVING                      Directory for saving model weights. Default is 'saved'.
  --seed SEED                          Random seed for PyTorch. Default is 17.
  --SE_results_dir SE_RESULTS_DIR      The directory for saving SentEval results. Default is 'SentEval_saved'.
  ```
Example of how to evaluate a model with the command line arguments:

```
$ python eval.py --model=LSTM
```

## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by Joy Crosbie - joy.crosbie@student.uva.nl
