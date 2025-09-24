import sys
import logging
import json
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



def run_experiment(best_dir="best_params", force_tune=False):
    """
    For each classifier:
      - If best-params JSON exists (and not force_tune), use it (no tuning).
      - Otherwise, run the tuner to create/update the JSON.
    Finally, run the stacking experiment (wp_experimenter).
    """
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        # (label, tuner_fn, expected_json_filename)
        ("SVM",                 tune_svm_wp,                 "svm_best_params.json"),
        ("Logistic Regression", tune_lr_wp,                  "lr_best_params.json"),
        ("Random Forest",       tune_rf_wp,                  "rf_best_params.json"),
        ("Decision Tree",       tune_dt_wp,                  "dt_best_params.json"),
        ("CatBoost",            lambda: tune_castboot_wp(task_type="CPU", n_trials=10),
                                 "cb_best_params.json"),
    ]

    for label, tuner, json_name in steps:
        print(f"[STEP] {label} ...")
        json_path = best_dir / json_name

        need_tune = force_tune or (not json_path.exists())
        if not need_tune:
            # Validate JSON readability (guards against truncated/corrupt files)
            try:
                with open(json_path, "r") as f:
                    _ = json.load(f)
                print(f"[OK] Using saved best params: {json_path.name}")
            except Exception as e:
                print(f"[WARN] Could not read {json_path.name} ({e}). Retuning {label}.")
                need_tune = True

        if need_tune:
            print(f"[TUNE] Running {label} tuner ...")
            tuner()
            if json_path.exists():
                print(f"[OK] Saved best params: {json_path.name}")
            else:
                print(f"[WARN] Expected {json_path.name} not found after tuning. "
                      f"Ensure your tuner writes this file.")

        print(f"[DONE] {label}")

    print("[STEP] Running stacking ensemble experiment ...")
    wp_experimenter()
    print("[DONE] Experiment finished successfully.")


if __name__ == "__main__":
    experiment_key = "experiment"
    # sys.stdout = LoggerWriter(logging.info, experiment_key)
    
    # Single entry point that makes the execution order explicit
    run_experiment()
