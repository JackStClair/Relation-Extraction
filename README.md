# Relation Extraction

[![Build Status](https://img.shields.io/badge/build-not%20configured-lightgrey)](https://github.com/JackStClair/Relation-Extraction/actions)
[![Test Coverage](https://img.shields.io/badge/coverage-not%20configured-lightgrey)](https://github.com/JackStClair/Relation-Extraction)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Neural network pipeline for multi-label relation extraction from conversational utterances.

## Project Overview

This project trains a PyTorch model to predict one or more relation labels for each utterance.
It combines:

- Character trigram features from `CountVectorizer`
- GloVe Twitter embeddings (`glove-twitter-200`)
- A multi-layer feedforward network with dropout

The script writes predictions to a CSV file in submission format.

## Repository Structure

- `run.py`: End-to-end training and inference script
- `requirements.txt`: Python dependency list
- `hw1_train.csv`: Training dataset
- `hw1_test.csv`: Test dataset
- `README.md`: Project documentation

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run training and generate predictions

```bash
python run.py hw1_train.csv hw1_test.csv submission.csv
```

### 3) Output

The command creates `submission.csv` with:

- `ID`
- `Core Relations`

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned package versions
