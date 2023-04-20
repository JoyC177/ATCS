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
* [Code Structure](#code-structure)
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

## Code Stucture
The code for the practical consists of the following files:
* train.py: To train one of the four models described above.
* senteval.py: To evaluate one of the four models described above on a subset of the SentEval tasks.
* encoderdecoder.py: Implementation for the Generic NLI training scheme from [Conneau et al. (2018)](https://arxiv.org/pdf/1705.02364.pdf).
* models.py: Adaptations of the Generic NLI training scheme for each of the four models described above.
* data.py: Processes the [SNLI dataset](https://nlp.stanford.edu/projects/snli/) and returns a dataloader.
* results.ipynb: Notebook containing demonstrations of results and error analyses.
	
## Setup

1. Create the environment:
```
conda env create -f env.yml
```

2. Install [SentEval](https://github.com/facebookresearch/SentEval) and follow the setup.

## Usage
### Training a model
The models can be trained with the following command line arguments:

```bash
usage: train.py [--model MODEL] [--train] [--lr LR] [--lr_decay LR_DECAY]
		    [--lr_decrease_factor LR_DECREASE_FACTOR] [--lr_threshold LR_THRESHOLD] 
		    [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS][--saving SAVING_DIR]
		    [--seed SEED] [--tensorboard_logs LOG_DIR] [--debug]

optional arguments:
  --model MODEL                   Name of the encoder model to use. Options: ['aweencoder', 'LSTM', 'BiLSTM', 'BiLSTMmax']. Default is 'aweencoder'.
  --train                         Force a training loop to overwrite existing models. Default is set to False.
  --learning_rate LR              Learning rate for the optimizer to use. Default is 0.1.
  --lr_decay LR_DECAY             Learning rate decay applied after each epoch. Default is 0.99.
  --lr_decrease LR_DECREASE       Factor to decrease learning rate with when val accuracy decreases. Default is 5.
  --lr_threshold LR_THRESHOLD     Learning rate threshold to stop training at. Default is 10e-5.
  --batch_size BATCH_SIZE         Minibatch size. Default is 64.
  --num_epochs NUM_EPOCHS         The maximum number of epochs to train for. Default is 20.
  --saving SAVING_DIR             Directory for saving model weights. Default is 'saved'.
  --seed SEED                     Random seed for PyTorch. Default is 17.
  --tensorboard_logs LOG_DIR      The directory for saving logs. Default is 'tensorboard_logs'.
  --debug                         Use 1% of data for debugging purposes. Default is set to False.
  ```
  
Example how to train a model from scratch with the command line arguments:

```
$ python train.py --model=LSTM --train=True
```
### Evaluating a model
The models can be evaluated on the SentEval tasks with the following command line arguments:

```bash
usage: senteval.py [--model MODEL] [--saving_dir SAVING_DIR]
		 [--seed SEED] [--SE_results_dir RESULTS_DIR]

optional arguments:
  --model MODEL                        Name of the encoder model to use. Options: ['aweencoder', 'LSTM', 'BiLSTM', 'BiLSTMmax']. Default is 'aweencoder'.
  --saving SAVING_DIR                  Directory for saving model weights. Default is 'saved'.
  --seed SEED                          Random seed for PyTorch. Default is 17.
  --SE_results_dir SE_RESULTS_DIR      The directory for saving SentEval results. Default is 'SentEval_saved'.
  ```
Example of how to evaluate a model with the command line arguments:

```
$ python senteval.py --model=LSTM
```

## Acknowledgements
- This project was created with the help of the Deep Learning I notebook tutorials: https://uvadlc-notebooks.readthedocs.io/en/latest/
- Insipration was taken from the NLP1 programming assignments and project.

## Contact
Created by Joy Crosbie - joy.crosbie@student.uva.nl
