from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .helper import load_xy_groups


# -------------- objective factory --------------
def make_objective(all_train_data, n_splits=4, seed=42):
    """
    Cria o objetivo do Optuna:
      - Para cada trial, avalia SVM(RBF) com StratifiedGroupKFold em CADA setup
      - Retorna a média das acurácias de CV entre os setups
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

        per_setup_acc = []
        for pack in all_train_data:
            X_tr, y_tr, g_tr = pack["X"], pack["y"], pack["groups"]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, random_state=seed))
            ])

            acc_mean = cross_val_score(
                pipe, X_tr, y_tr,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                groups=g_tr
            ).mean()
            per_setup_acc.append(acc_mean)

        # média global entre setups
        return float(np.mean(per_setup_acc))

    return objective


# -------------- main --------------
def main():
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1))  # setup_1 .. setup_10

    # Preloads train/test data from all setups
    all_train_data = []  
    all_test_data  = []  
    for i in setup_ids:
        root = base_dir / f"setup_{i}"
        train_csv = root / "train.csv"
        test_csv  = root / "test.csv"

        X_tr, y_tr, g_tr = load_xy_groups(train_csv)
        X_te, y_te, _    = load_xy_groups(test_csv)  

        all_train_data.append({"setup": i, "X": X_tr, "y": y_tr, "groups": g_tr})
        all_test_data.append({"setup": i, "X": X_te, "y": y_te})

    print(f"[INFO] Carregados {len(all_train_data)} setups para tuning global.")

    # Optuna: objective -> average CV accuracy among the 10 setups
    objective = make_objective(all_train_data, n_splits=4, seed=42)
    study = optuna.create_study(direction="maximize", study_name="svm_wp_sgkfold_all_setups")
    study.optimize(objective, n_trials=100)

    print("\n===== OPTUNA BEST (GLOBAL CV MEAN) =====")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"Global CV mean accuracy: {study.best_value:.4f}")

    # Train by setup in full train with the best global hyperparameters
    bp = study.best_params
    per_setup_acc = []

    print("\n===== TEST RESULTS PER SETUP =====")
    for pack_train, pack_test in zip(all_train_data, all_test_data):
        sid = pack_train["setup"]
        X_tr, y_tr = pack_train["X"], pack_train["y"]
        X_te, y_te = pack_test["X"], pack_test["y"]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=bp["C"],
                gamma=bp["gamma"],
                probability=True,
                random_state=42
            ))
        ])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        rep = classification_report(y_te, y_pred, digits=4)

        per_setup_acc.append(acc)
        print(f"\n--- setup_{sid} ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification Report:")
        print(rep)

    print("\n===== GLOBAL TEST SUMMARY =====")
    print(f"Mean accuracy over setups: {np.mean(per_setup_acc):.4f}")
    print(f"Std  accuracy over setups: {np.std(per_setup_acc):.4f}")

    # Save best params
    with open("best_params/svm_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("[INFO] Saved best params JSON.")


if __name__ == "__main__":
    main()
