# Breast Cancer Detection: Benign vs. Malignant

A machine learning project that classifies breast tissue histopathology images as benign or malignant. The main model is a Multilayer Perceptron (MLP), compared against Logistic Regression and Random Forest baselines.

## Dataset

**BreaKHis v1**: 7,909 microscopy images of breast tumor tissue at four magnification levels.

- Benign: 2,480 images
- Malignant: 5,429 images

The dataset is not included in this repo. Get it from the official source, the Laboratory of Vision, Robotics and Imaging (UFPR):

- Dataset page: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- Direct download: http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz

Extract the archive and place it so the folder structure looks like this:

```
project folder/
    breast_cancer_mlp.py
    BreakHisData/
        BreaKHis_v1/
            BreaKHis_v1/
                histology_slides/
                    breast/
                        benign/
                        malignant/
```

If your data lives somewhere else, change the `DATA_ROOT` line near the top of the script.

BreaKHis may be used for non commercial research if you cite the original paper:

> Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L. "A Dataset for Breast Cancer Histopathological Image Classification." IEEE Transactions on Biomedical Engineering (TBME), 63(7):1455-1462, 2016.

## Pipeline

1. Load every PNG, convert to grayscale, resize to 64x64, and flatten to 4,096 features
2. Normalize pixel values to the range 0 to 1
3. Oversample the benign class (with replacement) to fix the roughly 31/69 class imbalance
4. Split 80/20 into train and test sets, stratified by class
5. Standardize features with StandardScaler (fit on training data only)
6. Train and evaluate three models

## Models

| Model | Notes |
|---|---|
| MLP (primary) | Two hidden layers (256, 128), ReLU, L2 regularization (alpha=0.01), early stopping |
| Logistic Regression | Linear baseline, uses standardized features |
| Random Forest | 100 trees, uses raw normalized features (no scaling needed) |

## Evaluation

Each model is scored on accuracy, precision, recall, and F1. Recall on the malignant class matters most here, since a missed cancer (false negative) is the worst error. The script also produces confusion matrices and ROC curves with AUC for all three models.

## Requirements

- Python 3.8 or newer
- numpy
- pandas
- matplotlib
- pillow
- scikit-learn

Install them with:

```
pip install numpy pandas matplotlib pillow scikit-learn
```

## Usage

```
python breast_cancer_mlp.py
```

The MLP can take a few minutes to train. Plots open in windows during the run (close each one to continue) and are also saved to a `figures/` folder:

- `class_distribution.png`
- `sample_images.png`
- `mlp_learning_curve.png`
- `confusion_matrices.png`
- `roc_curves.png`

## Limitations

- Converting to grayscale throws away H&E staining color, which carries real diagnostic information.
- Oversampling with replacement duplicates training samples, which can raise overfitting risk.
- Flattening images loses spatial structure. A CNN trained on full resolution color images would be a better fit for production use.
