# anti-cheat-ml-models
A Machine Learning project aimed at detecting cheating behavior in online multiplayer video games. This work explores and compares different classification models: Random Forest, Gradient Boosting and XGBoost, under various hyperparameter optimization strategies, both with and without class balancing (SMOTE). The dataset is based on telemetry from Counter-Strike matches, as presented in [this paper](https://arxiv.org/html/2409.14830v1#S5) is available for download on [the figshare](https://figshare.com/s/b41992b81a480337cab8?file=48994417).

## Overview

This repository contains the full code and results for a university thesis project focused on building and evaluating machine learning models for automatic cheat detection. Three classification algorithms were compared:

- Random Forest
- Gradient Boosting
- XGBoost

Each was evaluated with and without SMOTE balancing, using:
- Manual search
- Grid Search
- Randomized Search
