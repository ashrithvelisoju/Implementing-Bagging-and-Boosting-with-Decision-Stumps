# Ensemble Learning: Implementing Bagging and Boosting with Decision Stumps 🌳📊

A comprehensive implementation and comparison of ensemble learning methods (Bagging and Boosting) using decision stumps as weak learners, demonstrated on the classic Titanic survival prediction dataset.

## Overview

This project provides both manual implementations and scikit-learn comparisons of two fundamental ensemble learning algorithms: **Bagging (Bootstrap Aggregating)** and **AdaBoost (Adaptive Boosting)**. The implementation demonstrates how these ensemble methods can improve prediction accuracy by combining multiple weak learners (decision stumps) into a stronger predictor.

The project serves as an educational tool for understanding ensemble learning concepts and provides practical implementations that can be extended for various machine learning applications.

## Key Features

### 🎯 Core Ensemble Methods
- **Manual Bagging Implementation**: Custom bootstrap aggregating algorithm with majority voting
- **Manual AdaBoost Implementation**: Custom adaptive boosting with weighted sampling and sequential learning
- **Scikit-learn Integration**: Professional implementation comparisons using BaggingClassifier and AdaBoostClassifier
- **Decision Stumps**: Single-level decision trees used as weak learners

### 📈 Advanced Analytics
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation
- **Performance Metrics**: Accuracy scores, confusion matrices, and classification reports
- **Feature Importance Analysis**: Identification of most influential features in ensemble models
- **Error Rate Tracking**: Visualization of AdaBoost error progression during training

### 🔍 Data Processing & Visualization
- **Automated Data Preprocessing**: Handling of categorical and numerical features with proper scaling
- **Missing Value Imputation**: Intelligent handling of missing data using median/mode strategies
- **Comprehensive Visualizations**: Model performance comparisons, error rate progressions, and feature importance plots
- **Statistical Analysis**: Detailed dataset exploration with descriptive statistics

### 🧪 Educational Components
- **Algorithm Transparency**: Step-by-step implementation details for understanding ensemble mechanics
- **Performance Comparisons**: Side-by-side evaluation of manual vs. professional implementations
- **Real-world Application**: Demonstrated on the Titanic dataset for practical relevance

## Prerequisites

### System Requirements
- **Python**: 3.7 or higher (3.8+ recommended)
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 1GB free space for dependencies and datasets
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Required Libraries
```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0  # For notebook execution
