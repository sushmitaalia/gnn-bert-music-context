# GNN-BERT Music Context Understanding

Course Project for CSE425 Neural Networks.

## Project Goal

The goal of this project is to build a hybrid GNN + BERT model for understanding musical context.

- BERT processes textual information such as music tags and captions.
- GNN processes structural relationships in music represented as graphs.
- Fusion models combine audio and text representations.
- The project also explores audio-text retrieval using contrastive learning.

## Project Tasks

The project is divided into four main tasks:

1. Text-based music tag classification using DistilBERT.
2. Graph-based genre classification using GraphSAGE with a CNN baseline.
3. GNN-BERT multimodal fusion using concatenation and cross-attention.
4. Audio-text retrieval using a contrastive GNN-BERT model.

## Datasets

The project uses the following datasets:

- MusicCaps
- FMA-small
- MagnaTagATune

Raw datasets are not included in the repository because of their large size.

## Project Structure

- `data/raw/` - raw dataset location
- `data/processed/` - processed audio features and graph samples
- `data/splits/` - train, validation, and test split information
- `notebooks/` - exploratory analysis and experiment notebooks
- `src/` - source code for preprocessing, models, training, and evaluation
- `results/` - metrics, plots, and experiment outputs
- `report/` - final project report


## Group Members

- Sushmita Alia — 23241026
- Md Minhaz Ibne Zaman — 23201357
- Rakib Shahriar — 23201145
