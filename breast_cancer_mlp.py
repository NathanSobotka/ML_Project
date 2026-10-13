"""
Breast Cancer Detection: Benign vs. Malignant

Dataset: BreaKHis v1, 7,909 histopathology microscopy images (4 magnification levels)
Primary Model: Multilayer Perceptron (MLP) with L2 regularization and early stopping
Comparison Models: Logistic Regression, Random Forest

Pipeline: Load PNGs, Resize 64x64, Grayscale, Flatten, Normalize,
          Oversample minority class, StandardScaler, Train/Evaluate
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

warnings.filterwarnings('ignore')

# Folder to save output figures (created next to this script)
os.makedirs('figures', exist_ok=True)

# You might have to change the path below to get everything to work!
DATA_ROOT = Path('BreakHisData/BreaKHis_v1/BreaKHis_v1/histology_slides/breast')

# Verify the path resolves before trying to load anything
if not DATA_ROOT.exists():
    raise FileNotFoundError(f'Dataset not found at: {DATA_ROOT.resolve()}\n'
                            'Make sure BreakHisData/ is in the same folder as this script.')

# Resize all images to IMG_SIZE x IMG_SIZE pixels before flattening.
# 64x64 gives 4,096 features, small enough for an MLP but large enough
# to retain basic structural information from the tissue images.
IMG_SIZE = 64

print(f'Dataset root: {DATA_ROOT.resolve()}')


# =============================================================================
# Section 1: Load and Explore Data
# =============================================================================

def load_images(data_root, img_size):
    # Walk through benign/ and malignant/ subdirectories.
    # Each PNG is opened, converted to grayscale, resized, and flattened to 1D.
    # Grayscale reduces features from 3*64*64=12,288 to 64*64=4,096
    # while keeping structural/texture info that distinguishes tumor types.
    # Note: H&E staining color does carry diagnostic info, so grayscale is a
    # simplification. A CNN with color would likely perform better.
    images = []
    labels = []
    skipped = 0  # track any corrupted/unreadable files

    for label_name, label_val in [('benign', 0), ('malignant', 1)]:
        class_dir = data_root / label_name
        for img_path in sorted(class_dir.rglob('*.png')):
            try:
                img = Image.open(img_path).convert('L')  # 'L' = grayscale
                img = img.resize((img_size, img_size))   # shrink to 64x64
                images.append(np.array(img, dtype=np.float32).flatten())
                labels.append(label_val)
            except Exception:
                # Skip files that cannot be opened (corrupted, incomplete download)
                skipped += 1

    if skipped > 0:
        print(f'WARNING: {skipped} images were skipped due to read errors.')

    return np.array(images), np.array(labels)


X, y = load_images(DATA_ROOT, IMG_SIZE)

print(f'Total images loaded : {len(y)}')
print(f'Feature vector size : {X.shape[1]} ({IMG_SIZE}x{IMG_SIZE} pixels)')
print(f'Benign  (label 0)   : {np.sum(y == 0)}')
print(f'Malignant (label 1) : {np.sum(y == 1)}')

# Visualize the raw class distribution to understand the imbalance
counts = [np.sum(y == 0), np.sum(y == 1)]
labels_names = ['Benign', 'Malignant']

plt.figure(figsize=(6, 4))
bars = plt.bar(labels_names, counts, color=['steelblue', 'tomato'], edgecolor='black')
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, count + 50,
             str(count), ha='center', fontsize=11)
plt.ylabel('Number of Images')
plt.title('Class Distribution (Before Balancing)')
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=120)
plt.show()

# Show sample images from each class so we can visually inspect the data.
# Displayed in color (opened fresh from disk) since grayscale loses stain info.
fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for row, (label_name, label_val) in enumerate([('Benign', 0), ('Malignant', 1)]):
    class_dir = DATA_ROOT / label_name.lower()
    sample_paths = sorted(class_dir.rglob('*.png'))[:5]
    for col, img_path in enumerate(sample_paths):
        img = Image.open(img_path).resize((200, 150))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(label_name, fontsize=12)

axes[0, 2].set_title('Sample Histopathology Images', fontsize=13)
plt.tight_layout()
plt.savefig('figures/sample_images.png', dpi=100)
plt.show()


# =============================================================================
# Section 2: Data Cleaning and Preprocessing
#
# 1. Normalize pixel values from [0, 255] to [0, 1] so large pixel values
#    don't dominate gradient updates.
# 2. Handle class imbalance. 2,480 benign vs 5,429 malignant (about 31/69).
#    A model trained on the raw split would be biased toward malignant.
#    Fix: oversample the benign class with replacement to match malignant.
# =============================================================================

# Without this, pixel values (0 to 255) are much larger than typical weight
# initializations, which can cause slow convergence or exploding/vanishing gradients.
X = X / 255.0

print(f'Pixel value range after normalization: [{X.min():.2f}, {X.max():.2f}]')

# Separate benign and malignant samples
X_benign = X[y == 0]
y_benign = y[y == 0]
X_malignant = X[y == 1]
y_malignant = y[y == 1]

# Oversample benign (minority class) with replacement to match malignant count.
# This prevents the model from simply predicting 'malignant' for everything
# and getting 69% accuracy without learning anything useful.
# Downside: duplicated samples can slightly increase overfitting risk.
X_benign_up, y_benign_up = resample(
    X_benign, y_benign,
    replace=True,
    n_samples=len(X_malignant),
    random_state=42
)

# Combine oversampled benign with original malignant
X_balanced = np.vstack([X_benign_up, X_malignant])
y_balanced = np.hstack([y_benign_up, y_malignant])

print(f'After oversampling, Benign: {np.sum(y_balanced == 0)}, Malignant: {np.sum(y_balanced == 1)}')
print(f'Total balanced dataset size: {len(y_balanced)}')


# =============================================================================
# Section 3: Train/Test Split and Feature Scaling
# =============================================================================

# 80/20 split with stratify=y to maintain the 50/50 class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    random_state=42,
    stratify=y_balanced
)

# StandardScaler: zero mean, unit variance per feature.
# Required for MLP and Logistic Regression, both are sensitive to feature scale.
# Fit ONLY on training data; apply same transform to test data.
# (Fitting on test data would leak information from the test set.)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Random Forest does NOT need scaling (tree splits are scale invariant),
# so we keep the raw normalized arrays for RF training.

print(f'Training set : {X_train.shape}')
print(f'Test set     : {X_test.shape}')
print(f'Train class split, Benign: {np.sum(y_train == 0)}, Malignant: {np.sum(y_train == 1)}')
print(f'Test  class split, Benign: {np.sum(y_test == 0)}, Malignant: {np.sum(y_test == 1)}')


# =============================================================================
# Section 4: MLP Classifier (Primary Model)
#
# Architecture: 4,096 inputs, 256 neurons, 128 neurons, 2 outputs
# Regularization: L2 weight decay (alpha=0.01) plus early_stopping.
# Overfitting is a real risk here: about 1M parameters on about 8,600 samples.
# =============================================================================

# ReLU avoids the vanishing gradient problem that sigmoid suffers from.
# alpha=0.01: L2 penalty on weights, penalizes large weights to reduce overfitting.
# early_stopping=True: hold out 10% of training data as a validation set;
#   stop training when validation score doesn't improve for 10 consecutive epochs.
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    alpha=0.01,
    max_iter=200,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
    verbose=False  # set True to watch per epoch loss
)

print('Training MLP... (may take a few minutes)')
mlp.fit(X_train_std, y_train)
print(f'Training finished at iteration: {mlp.n_iter_}')
print(f'Best validation score         : {mlp.best_validation_score_:.4f}')

# Plot training loss and validation accuracy over epochs.
# Overfitting sign: training loss keeps dropping but validation score stalls or drops.
# Underfitting sign: both curves plateau at a high loss / low accuracy early on.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(mlp.loss_curve_, color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('MLP Training Loss Curve')
ax1.grid(True)

ax2.plot(mlp.validation_scores_, color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('MLP Validation Accuracy Curve')
ax2.grid(True)

plt.tight_layout()
plt.savefig('figures/mlp_learning_curve.png', dpi=120)
plt.show()

# Evaluate MLP on the held out test set
y_pred_mlp = mlp.predict(X_test_std)
y_prob_mlp = mlp.predict_proba(X_test_std)[:, 1]  # probability of malignant

# In cancer diagnosis, RECALL (sensitivity) is the most critical metric.
# A false negative (missing cancer) is far more dangerous than a false positive.
print('=== MLP Results ===')
print(classification_report(y_test, y_pred_mlp, target_names=['Benign', 'Malignant']))


# =============================================================================
# Section 5: Comparison Models
#
# Logistic Regression: linear baseline, establishes a floor for performance.
# Random Forest: non linear ensemble of 100 trees, no feature scaling needed.
# =============================================================================

# Logistic Regression uses standardized features (scale sensitive)
# C=1.0: default L2 regularization strength (inverse, smaller = stronger)
# max_iter=1000: increased from default 100 to allow convergence on high dim data
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

print('Training Logistic Regression...')
lr.fit(X_train_std, y_train)

y_pred_lr = lr.predict(X_test_std)
y_prob_lr = lr.predict_proba(X_test_std)[:, 1]

print('=== Logistic Regression Results ===')
print(classification_report(y_test, y_pred_lr, target_names=['Benign', 'Malignant']))

# Random Forest uses raw normalized features (no scaling needed)
# n_estimators=100: 100 trees; more is more stable but slower
# n_jobs=-1: use all CPU cores to speed up training
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print('Training Random Forest...')
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print('=== Random Forest Results ===')
print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Malignant']))


# =============================================================================
# Section 6: Model Comparison
# =============================================================================

# Summary metrics table, all three models side by side
model_names = ['MLP', 'Logistic Regression', 'Random Forest']
y_preds = [y_pred_mlp, y_pred_lr, y_pred_rf]

rows = []
for name, y_pred in zip(model_names, y_preds):
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall (Sensitivity)': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

df_results = pd.DataFrame(rows).set_index('Model')
print(df_results.round(4).to_string())

# Confusion matrices for each model
# False Negative (FN): malignant predicted as benign, the most dangerous error type
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, name, y_pred in zip(axes, model_names, y_preds):
    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm, cmap='Blues')
    ax.set_title(name, fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malignant'])
    ax.set_yticklabels(['Benign', 'Malignant'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm[r, c], ha='center', va='center', fontsize=14,
                    color='white' if cm[r, c] > cm.max() / 2 else 'black')

plt.suptitle('Confusion Matrices', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=120)
plt.show()

# ROC curves: True Positive Rate vs False Positive Rate at every threshold.
# AUC closer to 1.0 means better discrimination. AUC of 0.5 is random guessing.
y_probs = [y_prob_mlp, y_prob_lr, y_prob_rf]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(7, 6))
for name, y_prob, color in zip(model_names, y_probs, colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random guessing (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curves: Model Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=120)
plt.show()


# =============================================================================
# Summary
#
# Recall is the most critical metric (a missed cancer is the worst outcome),
# precision matters because false positives trigger unnecessary treatment,
# F1 balances both, and AUC shows overall discriminative power.
#
# Limitations:
# - Grayscale conversion discards H&E staining color, which carries diagnostic info.
# - Oversampling with replacement duplicates training samples, which can
#   increase overfitting risk.
# - A CNN trained on full resolution color images would be better for production.
# =============================================================================
