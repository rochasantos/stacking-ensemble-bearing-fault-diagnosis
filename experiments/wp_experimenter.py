# train_stack_wp_fixedparams.py
# ---------------------------------------------------------
# Stacking (RF, DT, LR, SVM) -> Meta: LogisticRegression
# Hiperparâmetros fixados (fornecidos pelo usuário)
# Treina e avalia por setup (1..10) em Wavelet Package features.
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import json
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .helper import load_best_params

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ---------------------- dataset helpers ----------------------
def load_xy(csv_path):
    """
    Lê X (colunas 'f*') e y ('label' ou 'label_str').
    """
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with 'f' in {csv_path}")

    if "label" in df.columns:
        y = df["label"].values
    elif "label_str" in df.columns:
        y = df["label_str"].values
    else:
        raise ValueError(f"No 'label' or 'label_str' column in {csv_path}")

    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y


# ----- base learners (fixos) -----
def make_base_estimators():
    """
    Build the base models with the provided hyperparameters.
    """
    # ----- Logistic Regression -----
    lr_best_params = load_best_params("best_params/lr_best_params.json")
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_best_params))
    ])

    # ----- Random Forest -----
    rf_best_params = load_best_params("best_params/rf_best_params.json")
    rf = RandomForestClassifier(**{k.replace("model__", ""): v for k, v in rf_best_params.items()})

    # Decision Tree
    dt_best_params = load_best_params("best_params/dt_best_params.json")
    dt = DecisionTreeClassifier(**dt_best_params)

    # SVM (RBF)
    svm_best_params = load_best_params("best_params/svm_best_params.json")
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(**svm_best_params, probability=True))
    ])

    return [
        ("rf", rf),
        ("dt", dt),
        ("lr", lr_base),
        ("svm", svm),
    ]


def make_meta_learner():
    """
    Meta-model: Logistic Regression (fixed with the same hyperparameters provided).
    Since the meta-model inputs are base probabilities, a scaler is not necessary here.
    """
    meta = LogisticRegression(
        penalty="l2",
        C=5.928449559376798,
        solver="saga",
        max_iter=1179,
        random_state=42,
        multi_class="auto",
        n_jobs=-1
    )
    return meta


# ----- evaluation -----
def evaluate_model(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=4)

    print(f"\n===== {name} =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-macro   : {f1m:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(rep)
    return acc, f1m


# ---------------------- main ----------------------
def main():
    base_dir = Path("wp_features")
    setup_ids = list(range(1, 10 + 1))

    # Fixed meta-learner
    meta = make_meta_learner()

    # Stacking: uses predict_proba of the four base models
    stack = StackingClassifier(
        estimators=make_base_estimators(),
        final_estimator=meta,
        stack_method="predict_proba",   # alimenta o meta com probas dos bases
        passthrough=False,              # apenas saídas dos bases (sem features originais)
        n_jobs=-1,
        cv=5                             # CV interna para gerar out-of-fold probas dos bases
    )

    all_acc = []
    all_f1m = []

    for sid in setup_ids:
        print(f"\n========== Setup {sid} ==========")
        root = base_dir / f"setup_{sid}"
        X_train, y_train = load_xy(str(root / "train.csv"))
        X_test,  y_test  = load_xy(str(root / "test.csv"))

        # Treinar e avaliar apenas o ensemble final (você pode remover se quiser métricas dos bases)
        print("[INFO] Training STACKING (RF + DT + LR + SVM -> LR meta)…")
        stack.fit(X_train, y_train)
        acc, f1m = evaluate_model("STACKING (RF+DT+LR+SVM -> LR meta)", stack, X_test, y_test)

        all_acc.append(acc)
        all_f1m.append(f1m)

    print("\n===== GLOBAL SUMMARY (Stacking) =====")
    print(f"Mean Accuracy: {np.mean(all_acc):.4f}  |  Std: {np.std(all_acc):.4f}")
    print(f"Mean F1-macro: {np.mean(all_f1m):.4f}  |  Std: {np.std(all_f1m):.4f}")

    # (Opcional) salvar o modelo do último setup
    # import joblib
    # joblib.dump(stack, "stack_wp_fixedparams.joblib")
    # print("[INFO] Saved model to stack_wp_fixedparams.joblib")


if __name__ == "__main__":
    main()
