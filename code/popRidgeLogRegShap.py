# ============================================================
# Ridge log regresion for Valence (Fig 4)
#
# "Prefrontal neural geometry of associated cues guides learned motivated behaviors"
# Winke N, Luthi A, Herry C, Jercog D
# DOI: XXX
# 
# (github.com/djercog/WinkeEtAl-value-2025)
# ============================================================

import os
import numpy as np
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import shap
from shap.maskers import Partition

# Parameters
cv_folds = 5
lambda_ridge = 0.01  # Regularization strength (C = 1 / lambda)
max_iter = 1000

# Loop through files
for i in range(1, 11):       # dat1 to dat10
    j = 3 #Valence
    input_name = f'decoder_input_dat{i}_{j}.mat'
    output_name = f'decoder_output_logit_dat{i}_{j}.mat'

    print(f"\n Processing {input_name}...")

    # Load data
    data = sio.loadmat(input_name)
    X_all = data['datAll']                          # [nTrials, nCells, nTimeBins]
    y_all = data['labelsAll'].flatten()             # [nTrials]
    weights = data['trialWeights'].flatten()        # [nTrials]

    # Remove label==0 trials
    valid_mask = y_all != 0
    X_all = X_all[valid_mask, :, :]
    y_all = y_all[valid_mask]
    weights = weights[valid_mask]

    # Convert labels from -1/+1 → 0/1
    y_all = ((y_all + 1) / 2).astype(int)

    n_trials, n_cells, n_timebins = X_all.shape
    # Init outputs
    y_preds = [None] * n_timebins
    betas = [None] * n_timebins
    shap_vals = [None] * n_timebins
    for t in range(n_timebins):
        print(f"Times bin {t+1}/{n_timebins}")
        X = X_all[:, :, t]
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)  # z-score
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_pred_all = []
        shap_all_parts = []

        for train_idx, test_idx in kf.split(X):
            X_train, y_train = X[train_idx], y_all[train_idx]
            X_test = X[test_idx]
            w_train = weights[train_idx]

            # Ridge logistic regression
            model = LogisticRegression(
                penalty='l2',
                C=1.0 / lambda_ridge,
                solver='lbfgs',
                max_iter=max_iter,
                class_weight='balanced',
            )
            model.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model.predict_proba(X_test)[:, 1]  # prob of class 1
            y_pred_all.append(y_pred.astype(np.float32))

            # SHAP values for class 1 (positive)
            masker = Partition(X_train)
            explainer = shap.Explainer(model, masker, algorithm="linear")
            shap_values = explainer(X_test).values     # [2, nTest, nCells]

            if isinstance(shap_values, list) or shap_values.ndim == 3:
                shap_values = shap_values[1]  # take class 1 SHAP values

            shap_all_parts.append(shap_values.astype(np.float32))

        y_preds[t] = np.concatenate(y_pred_all, axis=0)
        shap_vals[t] = np.concatenate(shap_all_parts, axis=0)
        betas[t] = model.coef_.astype(np.float32)  # [1 x nCells]

    # Save
    sio.savemat(output_name, {
        'y_preds': y_preds,
        'shap_vals': shap_vals,
        'betas': betas
    })

print("done.")
