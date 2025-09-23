# tune_catboost_wp_sgkfold_all_setups.py
# ---------------------------------------------------------
# CatBoost + Optuna com StratifiedGroupKFold (4 folds)
# Tuning GLOBAL nos setups 1..10:
#   - Objetivo = média das acurácias de CV (por-setup) ao longo dos 10 setups
# Após o tuning, treina por-setup com os melhores hiperparâmetros globais
# e avalia no test de cada setup.
# CPU-safe: sem Poisson; 'subsample' só em Bernoulli/MVS (não em Bayesian).
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier


# -------------- helpers --------------
def _extract_acquisition_from_path(p: str):
    """
    Extrai o token de aquisição {acq} de {label}_{acq}_{mode}_{idx}.npy
    Ex.: ".../B_15_1_95.npy" -> 15
    """
    fname = Path(str(p)).name
    parts = fname.split("_")
    if len(parts) < 4:
        raise ValueError(f"Formato inesperado de filename para agrupar: {fname}")
    acq = parts[1]
    try:
        return int(acq)
    except ValueError:
        return acq  # fallback string


def load_xy_groups(csv_path: Path):
    """
    Carrega X, y, groups de um CSV:
      - features: colunas começando com 'f'
      - rótulos: 'label' (int) OU 'label_str' (mapeada para int)
      - grupos: aquisição extraída de 'path'
    """
    df = pd.read_csv(csv_path)

    if "path" not in df.columns:
        raise ValueError(f"CSV precisa conter a coluna 'path': {csv_path}")

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError(f"Nenhuma coluna de features iniciando em 'f' em {csv_path}")

    if "label" in df.columns:
        y = df["label"].values
    elif "label_str" in df.columns:
        mapping = {s: i for i, s in enumerate(sorted(df["label_str"].astype(str).unique()))}
        y = df["label_str"].map(mapping).values
    else:
        raise ValueError(f"CSV precisa ter 'label' ou 'label_str': {csv_path}")

    groups = df["path"].apply(_extract_acquisition_from_path).values
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y, groups


# -------------- CatBoost param builder (CPU-safe) --------------
def build_cat_params_from_trial(trial, task_type="CPU"):
    """
    Sugere hiperparâmetros CatBoost compatíveis:
      - CPU: sem Poisson; 'subsample' só em Bernoulli/MVS (não em Bayesian)
    """
    if task_type.upper() == "CPU":
        bootstrap_choices = ["Bayesian", "Bernoulli", "MVS"]
    else:
        bootstrap_choices = ["Bayesian", "Bernoulli", "Poisson", "MVS"]

    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 800),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),  # column subsampling
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", bootstrap_choices),
        "task_type": task_type,
        # "random_state": 42,
        # "verbose": 0,
        # loss_function será passado apenas no construtor (fora deste dict)
    }
    if params["bootstrap_type"] in ("Bernoulli", "Poisson", "MVS"):
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return params


def rebuild_params(best_dict, task_type="CPU"):
    """
    Reconstrói o dicionário de params de forma consistente (reaplica regra do subsample).
    """
    class _T:
        def __init__(self, d): self.d = d
        def suggest_int(self, n, a, b): return self.d[n]
        def suggest_float(self, n, a, b, log=False): return self.d[n]
        def suggest_categorical(self, n, opts): return self.d[n]
    return build_cat_params_from_trial(_T(best_dict), task_type=task_type)


# -------------- objective factory (global mean accuracy) --------------
def make_objective(all_train_data, n_splits=4, seed=42, task_type="CPU"):
    """
    For each trial:
    - builds a CatBoost with the proposed parameters
    - calculates the average CV accuracy (SGKFold) in EACH setup
    - returns the average accuracy between setups (maximize)
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = build_cat_params_from_trial(trial, task_type=task_type)

        per_setup_acc = []
        for pack in all_train_data:
            X_tr, y_tr, g_tr = pack["X"], pack["y"], pack["groups"]

            model = CatBoostClassifier(
                **params,
                loss_function="MultiClass",
            )

            acc_mean = cross_val_score(
                model, X_tr, y_tr,
                scoring="accuracy",
                cv=cv,
                n_jobs=-1,
                groups=g_tr
            ).mean()

            per_setup_acc.append(acc_mean)

        return float(np.mean(per_setup_acc))

    return objective


# -------------- main --------------
def main(task_type="CPU", n_trials=10):
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1))  

    # Preloads train/test data from all setups
    all_train_data = []  
    all_test_data  = []  
    for i in setup_ids:
        root = base_dir / f"setup_{i}"
        train_csv = root / "train.csv"
        test_csv  = root / "test.csv"

        X_tr, y_tr, g_tr = load_xy_groups(train_csv)
        X_te, y_te, _    = load_xy_groups(test_csv)  # groups não usados no teste

        all_train_data.append({"setup": i, "X": X_tr, "y": y_tr, "groups": g_tr})
        all_test_data.append({"setup": i, "X": X_te, "y": y_te})

    print(f"[INFO] Carregados {len(all_train_data)} setups para tuning global.")

    # Optuna: objective = average CV accuracy among the 10 setups
    objective = make_objective(all_train_data, n_splits=4, seed=42, task_type=task_type)
    study = optuna.create_study(direction="maximize", study_name="catboost_wp_sgkfold_all_setups")
    study.optimize(objective, n_trials=n_trials)

    print("\n===== OPTUNA BEST (GLOBAL CV MEAN) =====")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
    print(f"Global CV mean accuracy: {study.best_value:.4f}")

    # Train by setup in full train with the best global hyperparameters
    bp = rebuild_params(study.best_params, task_type=task_type)
    per_setup_acc = []

    print("\n===== TEST RESULTS PER SETUP =====")
    for pack_train, pack_test in zip(all_train_data, all_test_data):
        sid = pack_train["setup"]
        X_tr, y_tr = pack_train["X"], pack_train["y"]
        X_te, y_te = pack_test["X"], pack_test["y"]

        final = CatBoostClassifier(
            **bp,
            loss_function="MultiClass",
            verbose=0
        )
        
        final.fit(
            X_tr, y_tr,
            eval_set=(X_te, y_te),
            use_best_model=True,
            early_stopping_rounds=50
        )

        y_pred = final.predict(X_te).ravel()
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


if __name__ == "__main__":
    # CPU by default. If you want GPU and Poisson, call main(task_type="GPU")
    main(task_type="CPU", n_trials=50)
