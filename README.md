# LANTERN Health Prediction
Hierarchical temporal deep learning model with attention for longitudinal health-state prediction, calibration, and risk stratification.

This repository contains code for training and evaluating LANTERN: a Longitudinal Attribute-conditioned Neural Transition Estimation Recurrent Network for modeling health-state transitions in irregular longitudinal health data.

LANTERN predicts the next observed health state for each person-wave record using health history, elapsed time between visits, and demographic/health covariates. The model outputs a probability distribution over four health states:
```text
H = Healthy
M = Mild disability
S = Severe disability
D = Death
```
The predicted individual-level transition probabilities can be used for calibration analysis, risk stratification, and aggregation into transition matrices.

## Requirements

This project requires **Python 3.10** or later versions. Install the required packages with:
```bash
pip install -r requirements.txt
```

## Train and evaluate
python3 main.py \
    --train \
    --eval \
    --save-split-path splits/lantern_split.npz \
    --output-dir FINAL_RESULTS/LANTERN_FULL

## Evaluate from saved checkpoint
python3 main.py \
    --eval \
    --ckpt-path FINAL_RESULTS/LANTERN_FULL/<run_id>/<model_name>/best_model.pt \
    --output-dir FINAL_RESULTS/LANTERN_FULL

## Run ablations
python3 ablations.py \
    --main-script main.py \
    --skip-baselines \
    --split-path splits/lantern_split.npz \
    --output-root ABLATION_RUNS_LANTERN_final

## Data
The raw RAND HRS data file is not included in this repository.
The preprocessing code expects the RAND HRS file locally and creates the final modeling dataset used by the training script.

## Citation
coming soon...

## Paper
This repository accompanies the paper:
**A Longitudinal Attribute-Conditioned Neural Network for Modeling Health-State Transition Probabilities in Temporally Irregular Data: The LANTERN Framework**
Paper link coming soon.
