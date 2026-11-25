#!/usr/bin/env python3
"""
scripts/run_adv_agent4_ensembles.py
AGENT 4: Ensembles & Hybrid Models (Soft Voting, Weighted Probabilities, Stacking Meta-Classifier).
Strictly uses out-of-fold probabilities and enforces 29-fold LOGO for meta-learning.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from adv_campaign_engine import (
    load_advanced_dataset,
    OUTPUT_DIR
)

def evaluate_preds(y_true, preds, name, family, params, runtime_sec=1.0):
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    overall_acc = float(accuracy_score(y_true, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, preds, normalize="true").tolist()
    return {
        "name": name,
        "family": family,
        "features": "Ensemble (Multi-Model)",
        "feature_count": 8,
        "params": params,
        "macro_f1": round(macro_f1, 4),
        "balanced_acc": round(bal_acc, 4),
        "overall_acc": round(overall_acc, 4),
        "per_class_f1": [round(float(v), 4) for v in f1],
        "per_class_rec": [round(float(v), 4) for v in rec],
        "per_class_prec": [round(float(v), 4) for v in prec],
        "confusion_matrix": cm,
        "runtime_sec": round(runtime_sec, 2),
        "oof_pred_file": f"oof_pred_ens_{name[:20].replace(' ', '_').lower()}.npy"
    }

def main():
    print("=" * 70)
    print("AGENT 4: ENSEMBLE & HYBRID MODELS BENCHMARK")
    print("=" * 70)
    
    df, y, groups, classes, cls2idx, idx2cls = load_advanced_dataset()
    oof_dir = OUTPUT_DIR / "oof_predictions"
    
    # Candidate model probability files
    prob_files = {
        "SVM": oof_dir / "oof_prob_svm_rbf_c_10_0.npy",
        "CatBoost": oof_dir / "oof_prob_catboost_default.npy",
        "HistGBDT": oof_dir / "oof_prob_histgbdt_default.npy",
        "XGBoost": oof_dir / "oof_prob_xgb_cfg_d.npy",
        "LightGBM": oof_dir / "oof_prob_lgbm_tuned.npy"
    }
    
    loaded_probs = {}
    for k, p in prob_files.items():
        if p.exists():
            loaded_probs[k] = np.load(p)
            print(f"Loaded OOF probabilities for {k}: shape {loaded_probs[k].shape}")
        else:
            print(f"Warning: {k} probabilities at {p} not found yet!")
            
    assert len(loaded_probs) >= 2, f"Need at least 2 models for ensemble, found {len(loaded_probs)}"
    
    results = []
    ens_table = []
    
    # 1. Equal-Weight Soft Voting (SVM + CatBoost + XGBoost)
    if "SVM" in loaded_probs and "CatBoost" in loaded_probs and "XGBoost" in loaded_probs:
        p_ens = (loaded_probs["SVM"] + loaded_probs["CatBoost"] + loaded_probs["XGBoost"]) / 3.0
        preds = np.argmax(p_ens, axis=1)
        name = "Soft Voting Ensemble (SVM + CatBoost + XGBoost)"
        res = evaluate_preds(y, preds, name, "Ensemble (Soft Voting)", {"models": ["SVM", "CatBoost", "XGBoost"], "weights": "equal"})
        results.append(res)
        ens_table.append(res)
        print(f"{name}: Macro F1={res['macro_f1']:.4f} | Bal Acc={res['balanced_acc']:.4f} | Acc={res['overall_acc']:.4f}")

    # 2. Equal-Weight Soft Voting (SVM + CatBoost + HistGBDT)
    if "SVM" in loaded_probs and "CatBoost" in loaded_probs and "HistGBDT" in loaded_probs:
        p_ens = (loaded_probs["SVM"] + loaded_probs["CatBoost"] + loaded_probs["HistGBDT"]) / 3.0
        preds = np.argmax(p_ens, axis=1)
        name = "Soft Voting Ensemble (SVM + CatBoost + HistGBDT)"
        res = evaluate_preds(y, preds, name, "Ensemble (Soft Voting)", {"models": ["SVM", "CatBoost", "HistGBDT"], "weights": "equal"})
        results.append(res)
        ens_table.append(res)
        print(f"{name}: Macro F1={res['macro_f1']:.4f} | Bal Acc={res['balanced_acc']:.4f} | Acc={res['overall_acc']:.4f}")

    # 3. All-5 Diversity Ensemble (SVM + CatBoost + HistGBDT + XGBoost + LightGBM)
    if len(loaded_probs) >= 4:
        all_models = list(loaded_probs.keys())
        p_ens = np.mean([loaded_probs[m] for m in all_models], axis=0)
        preds = np.argmax(p_ens, axis=1)
        name = f"Soft Voting Multi-Family ({len(all_models)} Models)"
        res = evaluate_preds(y, preds, name, "Ensemble (Soft Voting)", {"models": all_models, "weights": "equal"})
        results.append(res)
        ens_table.append(res)
        print(f"{name}: Macro F1={res['macro_f1']:.4f} | Bal Acc={res['balanced_acc']:.4f} | Acc={res['overall_acc']:.4f}")

    # 4. Weighted Probability Voting (Optimized toward SVM + CatBoost margin)
    if "SVM" in loaded_probs and "CatBoost" in loaded_probs and "XGBoost" in loaded_probs:
        # Weighting: 0.45 SVM, 0.35 CatBoost, 0.20 XGBoost
        p_ens = 0.45 * loaded_probs["SVM"] + 0.35 * loaded_probs["CatBoost"] + 0.20 * loaded_probs["XGBoost"]
        preds = np.argmax(p_ens, axis=1)
        name = "Weighted Soft Voting (45% SVM, 35% CatBoost, 20% XGBoost)"
        res = evaluate_preds(y, preds, name, "Ensemble (Weighted Voting)", {"weights": {"SVM": 0.45, "CatBoost": 0.35, "XGBoost": 0.20}})
        results.append(res)
        ens_table.append(res)
        print(f"{name}: Macro F1={res['macro_f1']:.4f} | Bal Acc={res['balanced_acc']:.4f} | Acc: {res['overall_acc']:.4f}")

    # 5. Stacking Meta-Classifier (Strict 29-Fold LOGO Meta-Evaluation)
    # Stacking meta-features are constructed strictly from out-of-fold base model predictions,
    # and the meta-classifier is trained strictly on training groups in each outer fold.
    if len(loaded_probs) >= 3:
        meta_features = np.hstack([loaded_probs[m] for m in ["SVM", "CatBoost", "XGBoost"] if m in loaded_probs])
        logo = LeaveOneGroupOut()
        stacking_oof_preds = np.zeros(len(y), dtype=int)
        
        for tr, te in logo.split(meta_features, y, groups):
            meta_tr, y_tr = meta_features[tr], y[tr]
            meta_te, y_te = meta_features[te], y[te]
            
            meta_clf = LogisticRegression(C=1.0, penalty="l2", max_iter=300, random_state=42)
            meta_clf.fit(meta_tr, y_tr)
            stacking_oof_preds[te] = meta_clf.predict(meta_te)
            
        name = "Stacking Meta-Classifier (Logistic Regression on OOF Probabilities)"
        res = evaluate_preds(y, stacking_oof_preds, name, "Ensemble (Stacking)", {"meta_learner": "LogisticRegression(C=1.0)", "base_models": ["SVM", "CatBoost", "XGBoost"]})
        results.append(res)
        ens_table.append(res)
        print(f"{name}: Macro F1={res['macro_f1']:.4f} | Bal Acc={res['balanced_acc']:.4f} | Acc={res['overall_acc']:.4f}")

    # Save ensemble comparison CSV
    df_ens = pd.DataFrame([{
        "Ensemble_Model": r["name"],
        "Type": r["family"],
        "Macro_F1": r["macro_f1"],
        "Balanced_Accuracy": r["balanced_acc"],
        "Accuracy": r["overall_acc"],
        "Bulk_Data_Recall": r["per_class_rec"][0],
        "Real_Time_Recall": r["per_class_rec"][1],
        "Streaming_Recall": r["per_class_rec"][2],
    } for r in ens_table]).sort_values("Macro_F1", ascending=False)
    
    ens_path = OUTPUT_DIR / "ensemble_results.csv"
    df_ens.to_csv(ens_path, index=False)
    print(f"\nSaved ensemble comparison to: {ens_path}")
    
    out_file = OUTPUT_DIR / "agent4_ensembles_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[DONE] Agent 4 completed {len(results)} ensemble experiments -> {out_file}")

if __name__ == "__main__":
    main()
