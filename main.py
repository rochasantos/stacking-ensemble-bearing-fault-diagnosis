# logging
import sys
import logging
from src.utils import LoggerWriter
from pathlib import Path

from experiments import tune_svm_wp, tune_lr_wp, tune_rf_wp, tune_dt_wp, tune_castboot_wp, wp_experimenter


BEST_PARAMS = {
    "svm": Path("best_params/svm_best_params.json"),
    "lr":  Path("best_params/lr_best_params.json"),
    "rf":  Path("best_params/rf_best_params.json"),
    "dt":  Path("best_params/dt_best_params.json"),
    "cb":  Path("best_params/cb_best_params.json"),  # if you add CatBoost later
}

def ensure_params_exist():
    """Abort if any best-params JSON is missing (forces tuning to run first)."""
    missing = [name for name, p in BEST_PARAMS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "[ERROR] Missing best-params JSON(s): "
            + ", ".join(f"{m} -> {BEST_PARAMS[m]}" for m in missing)
            + ". You must run hyperparameter tuning before executing the experiment."
        )

def run_experiment():
    """Run experiments in a strict, readable order."""
    steps = [
        ("Tuning SVM", tune_svm_wp),
        ("Tuning Logistic Regression", tune_lr_wp),
        ("Tuning Random Forest", tune_rf_wp),
        ("Tuning Decision Tree", tune_dt_wp),
        ("Tuning CatBoost", lambda: tune_castboot_wp(task_type="CPU", n_trials=10)),
    ]

    for label, fn in steps:
        print(f"[STEP] {label} ...")
        fn()
        print(f"[OK] {label} completed.")

    print("[STEP] Validating best-params files ...")
    ensure_params_exist()
    print("[OK] All best-params JSON files are present.")

    print("[STEP] Running stacking ensemble experiment ...")
    wp_experimenter()
    print("[DONE] Experiment finished successfully.")

if __name__ == "__main__":
    experiment_key = "experiment"
    # sys.stdout = LoggerWriter(logging.info, experiment_key)
    
    # Single entry point that makes the execution order explicit
    run_experiment()