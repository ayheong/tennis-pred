# src/models/train.py
from pathlib import Path
import json
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    log_loss, roc_auc_score, accuracy_score,
    roc_curve, balanced_accuracy_score, brier_score_loss
)

from src.features.build_features import (
    load_matches, dataframe_to_examples,
    fit_label_encoders, save_encoders,
    NUM_DIFF_COLS, CAT_COLS
)

RANDOM_STATE = 42
SEEDS = [42, 7, 99, 1234, 2025]

if __name__ == "__main__":
    print("Loading matches...")
    df = load_matches("data/raw/atp_matches_202*.csv")
    print("Matches loaded:", df.shape)

    print("Building features...")
    X_raw, y, w = dataframe_to_examples(df, flip_prob=0.5, random_state=RANDOM_STATE, half_life_days=365)

    X_enc, encoders = fit_label_encoders(X_raw.copy())
    for c in CAT_COLS:
        if c in X_enc.columns:
            X_enc[c] = X_enc[c].astype("category")

    val_frac = 0.1
    cut = int(len(X_enc) * (1 - val_frac))
    X_train, y_train, w_train = X_enc.iloc[:cut], y.iloc[:cut], w.iloc[:cut]
    X_val,   y_val,   w_val   = X_enc.iloc[cut:], y.iloc[cut:], w.iloc[cut:]
    print(f"Sequential split → train: {len(X_train)}, val: {len(X_val)}")

    base_params = dict(
        objective="binary", metric="auc", n_estimators=2000,
        learning_rate=0.03488161148485931, num_leaves=181, max_depth=9,
        min_child_samples=13, subsample=0.8043814299639255, subsample_freq=1,
        colsample_bytree=0.6031707109518488, reg_alpha=0.00013001163094298563,
        reg_lambda=0.3364770049951767, min_split_gain=0.7891729810363337,
        max_bin=379, verbosity=-1, n_jobs=-1
    )

    models, val_preds_each = [], []
    for s in SEEDS:
        params = {**base_params, "random_state": s}
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric="auc",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        models.append(mdl)
        val_preds_each.append(mdl.predict_proba(X_val)[:, 1])

    preds = np.mean(val_preds_each, axis=0)

    fpr, tpr, thr = roc_curve(y_val, preds)
    j_scores = tpr - fpr
    best_thr = float(thr[j_scores.argmax()])

    yhat_05  = (preds >= 0.5).astype(int)
    yhat_opt = (preds >= best_thr).astype(int)

    metrics = {
        "logloss": float(log_loss(y_val, preds)),
        "auc": float(roc_auc_score(y_val, preds)),
        "accuracy@0.50": float(accuracy_score(y_val, yhat_05)),
        "accuracy@optJ": float(accuracy_score(y_val, yhat_opt)),
        "balanced_accuracy@optJ": float(balanced_accuracy_score(y_val, yhat_opt)),
        "brier": float(brier_score_loss(y_val, preds)),
        "best_threshold_optJ": best_thr,
        "best_iteration_mean": int(np.mean([getattr(m, "best_iteration_", 0) or 0 for m in models])),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_models": len(models),
    }
    print("Validation metrics (ensemble):", metrics)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(models, "models/lgb_ensemble.pkl")
    save_encoders(encoders, "models/encoders.json")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open("models/schema.json", "w") as f:
        json.dump({"numeric": NUM_DIFF_COLS, "categorical": CAT_COLS}, f, indent=2)
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": best_thr}, f, indent=2)

    print("Artifacts saved to models/")
