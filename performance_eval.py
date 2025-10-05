# performance_eval.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score,
    accuracy_score, confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt

# --- 1. Synthetic Data Generation (Replace with actual detection results) ---
np.random.seed(42) 
N = 1000 # Number of test images

# True binary labels (0: Watermark Absent, 1: Watermark Present)
y_true = np.random.randint(0, 2, N)

# Hypothetical predicted probability scores (higher score = more confident of watermark presence)
# Existing Model (Lower separation, simulating lower robustness/security)
noise_dwt = np.random.normal(0, 0.2, N)
scores_dwt_svd = np.clip(y_true * 0.8 + (1 - y_true) * 0.2 + noise_dwt, 0.05, 0.95)

# Proposed Model (Higher separation, simulating superior robustness/security)
noise_proposed = np.random.normal(0, 0.1, N)
scores_proposed = np.clip(y_true * 0.9 + (1 - y_true) * 0.1 + noise_proposed, 0.05, 0.99)

models = {
    'Existing Model (DWT-SVD)': scores_dwt_svd,
    'Proposed Model (LWT-SVD-CNN-RL-QNN)': scores_proposed
}

# --- 2. Calculate All Metrics ---
threshold = 0.5 
metrics_data = []
auc_scores = {}
pr_auc_scores = {}

for name, scores in models.items():
    # Binarize scores to get predictions for discrete metrics
    y_pred = (scores >= threshold).astype(int)
    
    # Calculate Core Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix (TP, TN, FP, FN counts)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() 
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    auc_scores[name] = roc_auc
    
    # Calculate AUPRC
    precision_p, recall_p, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall_p, precision_p)
    pr_auc_scores[name] = pr_auc
    
    metrics_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'AUC': roc_auc,
        'AUPRC': pr_auc,
        'Computation Time (s)': 'TBD',
        'True Positive (TP)': tp,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn,
        'True Negative (TN)': tn,
    })

metrics_df = pd.DataFrame(metrics_data).round(4)

# --- 3. Plotting ROC Curve ---
plt.figure(figsize=(8, 6))
for name in models.keys():
    fpr, tpr, _ = roc_curve(y_true, models[name])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_scores[name]:.3f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Recall')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# --- 4. Plotting Precision-Recall Curve ---
plt.figure(figsize=(8, 6))
for name in models.keys():
    precision_p, recall_p, _ = precision_recall_curve(y_true, models[name])
    plt.plot(recall_p, precision_p, label=f'{name} (AUPRC = {pr_auc_scores[name]:.3f})')

baseline = np.sum(y_true) / len(y_true)
plt.axhline(baseline, color='gray', linestyle='--', label=f'Baseline (P={baseline:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# --- 5. Print Final Results Table ---
print("\n" + "="*70)
print("FINAL PERFORMANCE METRICS TABLE (Threshold = 0.5)")
print("="*70)
print(metrics_df[['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'AUPRC', 'Computation Time (s)']].to_markdown(index=False))

print("\nConfusion Matrices:")
print(metrics_df[['Model', 'True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']].to_markdown(index=False))
