# Dataset Poisoning Detection

This project provides a comprehensive pipeline for simulating and detecting dataset poisoning attacks on the MNIST dataset. It implements various attack strategies and multiple state-of-the-art detection methodologies to identify malicious samples.

## Overview

Dataset poisoning is a security threat where an attacker injects malicious data into a training set to compromise the integrity of a machine learning model. This project explores:
- **Attack Simulation**: Implementing common poisoning techniques like Label Flipping and Backdoor attacks.
- **Detection Methodologies**: Evaluating different statistical and deep learning approaches to identify poisoned samples.

## Attack Strategies

1.  **Label Flip Attack**: Randomly changes the labels of a subset of training samples to a different class.
2.  **Backdoor Attack**: Injects a specific trigger pattern into samples and changes their target label to a fixed attacker-chosen class.

## Detection Methodologies

The pipeline employs the following detection techniques:

-   **Isolation Forest**: An unsupervised anomaly detection algorithm that isolates outliers by randomly sub-sampling and constructing trees.
-   **Autoencoder Detector**: Trains a neural network to reconstruct inputs; samples with high reconstruction error are flagged as potential poison.
-   **Influence Functions (TracIn Proxy)**: Estimates the influence of each training sample on the model's loss; highly "influential" (or high-loss) samples often indicate poisoning.
-   **Trust Score**: Measures the agreement between a classifier and a modified k-Nearest Neighbors (kNN) approach to determine sample reliability.

## Project Structure

```text
dataset_poison_detection/
├── main.py                 # Pipeline orchestrator
├── scripts/
│   ├── poison_attack.py     # Attack simulation
│   ├── feature_profiling.py  # Data distribution analysis
│   ├── isolation_forest_detector.py
│   ├── autoencoder_detector.py
│   ├── influence_functions.py
│   ├── trust_score.py
│   └── dataset_utils.py     # Helper classes for data handling
└── data/
    ├── raw/                # MNIST source data
    └── processed/          # Poisoned datasets and detection results
```

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

The entire pipeline—from data download and attack simulation to detection and evaluation—can be run using the orchestrator:

```bash
python dataset_poison_detection/main.py
```

Individual stages can also be run independently from the `scripts/` directory:

1.  **Attack**: `python dataset_poison_detection/scripts/poison_attack.py`
2.  **Detection**: Run any of the detector scripts (e.g., `python dataset_poison_detection/scripts/autoencoder_detector.py`)

## Results and Visualization

-   **Metrics**: Precision and Recall for each detection method are printed to the console.
-   **Visualizations**: Feature distribution plots and detector flags are saved in `data/processed/`.

---
*Created as part of the Dataset Poisoning Research project.*
