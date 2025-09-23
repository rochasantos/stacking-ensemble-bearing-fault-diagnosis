# tune_stacking_wp.py  (CPU-safe CatBoost inside Stacking)
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix

_HAS_CATBOOST = True
try:
    from catboost import CatBoostClassifier
except Exception:
    _HAS_CATBOOST = False
    warnings.warn("[WARN] CatBoost not installed; stacking will run without CatBoost.")


def load_xy(p):
    df = pd.read_csv(p)
    feat = [c for c in df.columns if c.startswith('f')]
    if not feat:
        raise ValueError(f"No feature columns starting with 'f' in {p}")
    y = df['label'].values if 'label' in df.columns else df['label_str'].values
    X = df[feat].to_numpy(dtype=np.float32)
    return X, y


def _catboost_from_trial(trial, task_type="CPU"):
    """
    Build a CPU-safe CatBoost with bootstrap/subsample consistency:
      - CPU: bootstrap_type in ["Bayesian","Bernoulli","MVS"]
      - subsample only if bootstrap_type in ["Bernoulli","MVS"]
    """
    if not _HAS_CATBOOST:
        return None

    # bootstrap options safe for CPU
    bootstrap_type = trial.suggest_categorical("cb_bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    # common search space
    depth = trial.suggest_int("cb_depth", 4, 10)
    learning_rate = trial.suggest_float("cb_lr", 1e-3, 0.3, log=True)
    # l2_leaf_reg = trial.suggest_float("cb_l2", 1e-3, 10.0, log=True)
    # n_estimators = trial.suggest_int("cb_n_estimators", 200, 800)
    # rsm = trial.suggest_float("cb_rsm", 0.6, 1.0)

    # build kwargs safely
    cb_kwargs = dict(
        depth=depth,
        learning_rate=learning_rate,
        # l2_leaf_reg=l2_leaf_reg,
        # n_estimators=n_estimators,
        # rsm=rsm,
        bootstrap_type=bootstrap_type,
        task_type=task_type,         # CPU safe
        loss_function="MultiClass",  # set only here
        random_state=42,
        verbose=0
    )

    if bootstrap_type in ("Bernoulli", "MVS"):
        cb_kwargs["subsample"] = trial.suggest_float("cb_subsample", 0.6, 1.0)
    # If Bayesian: DO NOT set subsample (avoid the exact error you saw)

    return CatBoostClassifier(**cb_kwargs)


def make_bases(trial):
    # Decision Tree
    dt = DecisionTreeClassifier(
        criterion=trial.suggest_categorical("dt_criterion", ["gini", "entropy", "log_loss"]),
        max_depth=trial.suggest_int("dt_max_depth", 5, 40),
        min_samples_split=trial.suggest_int("dt_min_split", 2, 20),
        min_samples_leaf=trial.suggest_int("dt_min_leaf", 1, 10),
        random_state=42
    )

    # Logistic Regression (with scaling)
    lr_C = trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
    lr_solver = trial.suggest_categorical("lr_solver", ["lbfgs", "saga"])
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=lr_C, solver=lr_solver, max_iter=3000, random_state=42))
    ])

    # SVM (with scaling)
    svm_C = trial.suggest_float("svm_C", 1e-2, 1e2, log=True)
    svm_gamma = trial.suggest_float("svm_gamma", 1e-4, 1e1, log=True)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=svm_C, gamma=svm_gamma, probability=True, random_state=42))
    ])

    bases = [("dt", dt), ("lr", lr), ("svm", svm)]

    # CatBoost (optional)
    if _HAS_CATBOOST:
        cb = _catboost_from_trial(trial, task_type="CPU")
        bases.append(("catboost", cb))

    return bases


def cv_f1_macro_stack(estimator, X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, scoring="f1_macro", cv=skf, n_jobs=-1)
    return float(scores.mean())


def main(n_stp, n_trials=60):
    root = f"wp_features/setup_{n_stp}"
    Xtr, ytr = load_xy(f"{root}/train.csv")
    Xte, yte = load_xy(f"{root}/test.csv")

    def objective(trial):
        estimators = make_bases(trial)

        meta_C = trial.suggest_float("meta_C", 1e-3, 1e2, log=True)
        meta = LogisticRegression(C=meta_C, max_iter=4000, random_state=42)

        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1
        )
        return cv_f1_macro_stack(clf, Xtr, ytr)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print("\n===== STACKING BEST PARAMS =====")
    print(best)

    # Rebuild models with the best params (re-apply the same safe logic):
    class _T:
        def __init__(self, d): self.d = d
        def suggest_int(self, n, a, b): return self.d[n]
        def suggest_float(self, n, a, b, log=False): return self.d[n]
        def suggest_categorical(self, n, opts): return self.d[n]

    estimators = make_bases(_T(best))
    meta = LogisticRegression(C=best["meta_C"], max_iter=4000, random_state=42)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )
    stack.fit(Xtr, ytr)
    ypred = stack.predict(Xte)

    print("\n===== STACKING TEST RESULTS =====")
    print("Accuracy:", accuracy_score(yte, ypred))
    print("F1-macro:", f1_score(yte, ypred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(yte, ypred))
    print("\nClassification Report:\n", classification_report(yte, ypred, digits=4))


if __name__ == "__main__":
    for n in range(1, 11):
        print(f"\n========== Setup {n} ==========")
        main(n, n_trials=80)
