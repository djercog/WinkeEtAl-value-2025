# ============================================================
# Ridge regresion for Salience and Value (Fig 4)
#
# "Prefrontal neural geometry of associated cues guides learned motivated behaviors"
# Winke N, Luthi A, Herry C, Jercog D. 
# DOI: XXX
# 
# github.com/djercog/WinkeEtAl-value-2025
# ============================================================


# C:\Users\dani\Documents\MATLAB
import os
import numpy as np
import scipy.io as sio
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import shap
from shap.maskers import Partition

# Parameters
cv_folds = 5
lambda_ridge = 0.01

# Loop through files
for i in range(1, 11):       # dat1 to dat10
    for j in range(1, 3):    # _1 to _2 (salience & value)
        input_name = f'decoder_input_dat{i}_{j}.mat'
        output_name = f'decoder_output_dat{i}_{j}.mat'

        
        print(f"Procesing {input_name}...")

        # Load data
        data = sio.loadmat(input_name)
        X_all = data['datAll']                          # [nTrials, nCells, nTimeBins]
        y_all = data['labelsAll'].flatten()             # [nTrials]
        weights = data['trialWeights'].flatten()        # [nTrials]

        n_trials, n_cells, n_timebins = X_all.shape

        y_preds = []
        betas = []
        shap_vals = []

        for t in range(n_timebins):
            print(f"  Times bin {t+1}/{n_timebins}")
            X = X_all[:, :, t]
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)  # z-score

            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            y_pred_all = []        # collect per-fold
            shap_all_parts = []    # collect per-fold
            
            for train_idx, test_idx in kf.split(X):
                X_train, y_train = X[train_idx], y_all[train_idx]
                X_test = X[test_idx]
                w_train = weights[train_idx]
            
                model = Ridge(alpha=lambda_ridge, fit_intercept=True)
                model.fit(X_train, y_train, sample_weight=w_train)
                y_pred = model.predict(X_test)
            
                masker = Partition(X_train)
                explainer = shap.Explainer(model, masker, algorithm="linear")
                shap_vals_test = explainer(X_test).values
            
                if isinstance(shap_vals_test, list):
                    shap_vals_test = shap_vals_test[0]
            
                y_pred_all.append(y_pred.astype(np.float32))         # (n_test,)
                shap_all_parts.append(shap_vals_test.astype(np.float32))  # (n_test, n_cells)
            
            # After folds, concatenate predictions and SHAPs
            y_preds.append(np.concatenate(y_pred_all, axis=0))
            shap_vals.append(np.concatenate(shap_all_parts, axis=0))
            betas.append(model.coef_.astype(np.float32))  # last model's weights

        # Save results
        sio.savemat(output_name, {
            'y_preds': np.array(y_preds, dtype=object),
            'betas': np.array(betas, dtype=object),
            'shap_vals': np.array(shap_vals, dtype=object)
        })

print("done.")
