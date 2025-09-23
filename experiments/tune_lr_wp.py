from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .helper import load_xy_groups


# ---------------------- objective factory ----------------------
def make_objective(all_train_data, n_splits=4, seed=42):
    """
    Create an Optuna objective:
      - Build LR pipeline (StandardScaler -> LogisticRegression)
      - Compute CV accuracy per setup using StratifiedGroupKFold(groups)
      - Return the MEAN accuracy across setups (maximize)
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        penalty = trial.suggest_categorical("penalty", ["l2", None])
        # if penalty == "none", C is ignored
        if penalty != "none":
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        max_iter = trial.suggest_int("max_iter", 500, 5000)

        per_setup_scores = []
        for pack in all_train_data:
            X_tr, y_tr, g_tr = pack["X"], pack["y"], pack["groups"]

            if penalty == "none":
                lr = LogisticRegression(
                    penalty="none",
                    solver=solver,
                    max_iter=max_iter,
                    random_state=42,
                    multi_class="auto"
                )
            else:
                lr = LogisticRegression(
                    penalty="l2",
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=42,
                    multi_class="auto"
                )

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", lr)
            ])

            acc_mean = cross_val_score(
                pipe, X_tr, y_tr,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                groups=g_tr
            ).mean()

            per_setup_scores.append(acc_mean)

        # Mean CV accuracy across all setups
        return float(np.mean(per_setup_scores))

    return objective


# ---------------------- main ----------------------
def main():
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1)) 

    # ---------- preload TRAIN/TEST for all setups ----------
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

    print(f"[INFO] Loaded {len(all_train_data)} setups for global tuning.")

    # ---------- Optuna global study ----------
    objective = make_objective(all_train_data, n_splits=4, seed=42)
    study = optuna.create_study(direction="maximize", study_name="lr_wp_sgkfold_all_setups")
    study.optimize(objective, n_trials=60)

    print("\n===== OPTUNA BEST (GLOBAL CV MEAN) =====")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"Global CV mean accuracy: {study.best_value:.4f}")

    # ---------- Train best model per setup and evaluate ----------
    bp = study.best_params
    per_setup_acc = []

    print("\n===== TEST RESULTS PER SETUP =====")
    for pack_train, pack_test in zip(all_train_data, all_test_data):
        sid = pack_train["setup"]
        X_tr, y_tr = pack_train["X"], pack_train["y"]
        X_te, y_te = pack_test["X"], pack_test["y"]

        # rebuild the best LR respecting 'penalty' logic
        if bp["penalty"] == "none":
            lr = LogisticRegression(
                penalty="none",
                solver=bp["solver"],
                max_iter=bp["max_iter"],
                random_state=42,
                multi_class="auto"
            )
        else:
            lr = LogisticRegression(
                penalty="l2",
                C=bp["C"],
                solver=bp["solver"],
                max_iter=bp["max_iter"],
                random_state=42,
                multi_class="auto"
            )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", lr)
        ])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred)
        report = classification_report(y_te, y_pred, digits=4)

        per_setup_acc.append(acc)
        print(f"\n--- setup_{sid} ---")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        print("Classification Report:")
        print(report)

    print("\n===== GLOBAL TEST SUMMARY =====")
    print(f"Mean accuracy over setups: {np.mean(per_setup_acc):.4f}")
    print(f"Std  accuracy over setups: {np.std(per_setup_acc):.4f}")

    # Save best params
    with open("best_params/lr_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("[INFO] Saved best params JSON.")


if __name__ == "__main__":
    main()
