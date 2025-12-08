# MABe Mouse Behavior Detection Challenge
### AAI-3303 Course Project - Machine Learning I

**Team Members:** Jeffrey Gregory, Mataeo Anderson  
**Instructor:** Dr. Ahmed Butt  

---

## Project Overview

This repository contains the code and analysis for our team's submission to the **2025 MABe (Multi-Agent Behavior) Challenge** on Kaggle. The goal of this competition is to detect and classify social behaviors in mice using tracking data.

The challenge involves analyzing high-dimensional tracking data (keypoints of mice) to predict specific social actions over time. This project served as our capstone work for the AAI-3303 Machine Learning I course, allowing us to explore advanced techniques in time-series classification and graph neural networks.

## Repository Structure

The repository is organized as follows:

*   **`CTRGCN-Submission-Notebook-EDU.ipynb`**: The primary educational scaffold for our **CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network)** model. This notebook outlines the pipeline for data loading, skeleton definition, model architecture, and submission.
*   **`mabe-comp-eda-notebook.ipynb`**: Exploratory Data Analysis (EDA) notebook. Contains code for inspecting the dataset structure, counting features, and visualizing tracking data.
*   **`social-action-recognition-in-mice-xgboost.ipynb`**: A baseline attempt using **XGBoost** for action recognition.
*   **`mabe-nearest-neighbors-the-original-ambrosm.ipynb`**: A baseline attempt using a **Nearest Neighbors** approach.
*   **`CTR-GCN-Models/`**: Directory containing model definitions and experiments related to CTR-GCN.
*   **`ST-GCN-Models/`**: Directory containing experiments with **ST-GCN (Spatial-Temporal Graph Convolutional Networks)**.
*   **`Milestone1_Deliverables/`**: Documentation and artifacts submitted for the first project milestone.
*   **`*.py` Scripts**: Various Python scripts (`CTRGCN-model-baseline.py`, etc.) used for offline training and testing of the GCN models.

## Methodology

Our approach evolved from simple baselines to more complex deep learning architectures suited for skeletal data.

### 1. Exploratory Data Analysis (EDA)
We started by analyzing the MABe dataset to understand the structure of the tracking data (keypoints, frames, IDs) and the distribution of the target behavior classes.

### 2. Baseline Models
*   **Nearest Neighbors & XGBoost:** A combination of others on Kaggle and ourselves implemented these standard machine learning models to establish a performance baseline and understand the feature importance of the raw tracking data.

### 3. Graph Convolutional Networks (GCN)
Recognizing that the mice can be modeled as "skeletons" (graphs of connected joints), we explored GCNs to capture the spatial relationships between body parts and the temporal dynamics of their movement.
*   **ST-GCN:** Spatial-Temporal GCNs were investigated for their ability to learn from graph sequences.
*   **CTR-GCN:** We focused heavily on **CTR-GCN**, a state-of-the-art architecture for skeleton-based action recognition, attempting to adapt it to the multi-agent context of the MABe challenge.

## Usage

### Prerequisites
The code is designed to run primarily in a **Kaggle Notebook** environment, where the MABe dataset is available.

**Key Dependencies:**
*   Python 3.x
*   PyTorch
*   Pandas, NumPy
*   Scikit-learn
*   Tqdm

### Running the Models
1.  **EDA:** Open `mabe-comp-eda-notebook.ipynb` to view the data analysis.
2.  **CTR-GCN:** The `CTRGCN-Submission-Notebook-EDU.ipynb` is designed to be run in "dev" mode for testing or "submit" mode for generating predictions. It requires pre-trained model weights (if running inference) or can be adapted for training said weights.

## 📝 Reflection

This project was a significant learning experience in handling complex, real-world biological data. While we faced challenges in training a highly competitive model due to the complexity of the multi-agent interactions and computational constraints, we gained valuable insights into:
*   Working with **Graph Neural Networks** for skeletal data.
*   Managing large-scale datasets and Kaggle competition workflows.
*   The importance of robust cross-validation and feature engineering in time-series tasks.

---
*This project was developed for educational purposes as part of the AAI-3303 course at OUPI.*
