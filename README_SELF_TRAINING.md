# Self-Improving Training Pipeline Instructions

## Overview
This document provides step-by-step instructions for using the self-improving training pipeline for the AquaChess project. The pipeline is designed to enhance the performance of the chess engine through iterative training and evaluation.

## Step 1: Set Up the Environment
Ensure you have the following dependencies installed:
- Python 3.8+
- Chess libraries (e.g., python-chess)
- Any additional libraries specific to AquaChess

### Example Command:
```bash
pip install -r requirements.txt
```

## Step 2: Configure Training Parameters
Edit the configuration file `config.yml` to set your desired training parameters. Key parameters include:
- `num_games`: Number of games to simulate in each training iteration.
- `elo_threshold`: Minimum Elo rating to apply the improvements.

### Example Configuration:
```yaml
num_games: 1000
elo_threshold: 2200
```

## Step 3: Run the Training Pipeline
Execute the training script to initiate the training pipeline. 

### Example Command:
```bash
python train.py
```

## Expected Results
After executing the training pipeline, you can expect the following Elo gains based on the number of games played:
- **100 games**: Expected Elo Gain: +50
- **500 games**: Expected Elo Gain: +200
- **1000 games**: Expected Elo Gain: +400

## Step 4: Evaluate the Results
Once training is complete, evaluate the performance by running the evaluation script:

### Example Command:
```bash
python evaluate.py
```

## Conclusion
Following these steps will help you effectively utilize the self-improving training pipeline in the AquaChess project, leading to improved performance metrics and Elo gains!