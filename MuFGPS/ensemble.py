# [ADDED] """
# [ADDED] ensemble.py
# [ADDED] 
# [ADDED] This script implements the stacking-based ensemble classifier for the MuFGPS framework.
# [ADDED] It loads pre-computed sequence, secondary structure, and GAT-based structural features,
# [ADDED] applies SMOTE oversampling on the training set, trains a stacking ensemble
# [ADDED] (Random Forest + XGBoost + LightGBM with logistic regression as meta-learner),
# [ADDED] evaluates the model, and saves metrics and ROC/PR curves.
# [ADDED] """

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from config import (
    BASIC_FEATURES_CSV, SEC_FEAT_CSV, GAT_FEAT_CSV, SPLIT_CSV,
    MERGED_TRAIN_CSV, MERGED_TEST_CSV, RESULTS_DIR,
    SMOTE_PARAMS, RF_PARAMS, XGB_PARAMS, LGB_PARAMS, LOGREG_PARAMS, RANDOM_SEED
)

def prepare_data():
      """
    Prepare merged training and test feature tables.

    This function:
        1. Loads basic sequence-level features, DSSP-based secondary structure features,
           and GAT-based structural embeddings.
        2. Merges them by protein ID.
        3. Splits the merged table into train / test according to SPLIT_CSV.
        4. Fills missing numeric values with column-wise medians.
        5. Saves the merged train/test tables for reproducibility.

    Returns
    -------
    train : pandas.DataFrame
        Merged training set with columns [id, label, feature_1, ..., feature_n].
    test : pandas.DataFrame
        Merged test set with the same structure as `train`.
    """
    base = pd.read_csv(BASIC_FEATURES_CSV)
    sec  = pd.read_csv(SEC_FEAT_CSV)
    gat  = pd.read_csv(GAT_FEAT_CSV)
    assert {"id","label"}.issubset(base.columns)
    # 只保留 id,label + 其他特征
    keep = [c for c in base.columns if c not in {"split"}]
    base = base[keep]

    # 合并
    m = base.merge(sec, on="id", how="left").merge(gat.drop(columns=["label"]), on="id", how="left")
    # 缺失填充
    m = m.fillna(m.median(numeric_only=True))

    split = pd.read_csv(SPLIT_CSV)
    tr_ids = set(split[split["split"]=="train"]["id"])
    te_ids = set(split[split["split"]=="test"]["id"])
    train = m[m["id"].isin(tr_ids)].copy()
    test  = m[m["id"].isin(te_ids)].copy()

    # 保存一下拼接结果
    train.to_csv(MERGED_TRAIN_CSV, index=False)
    test.to_csv(MERGED_TEST_CSV, index=False)
    return train, test

def split_xy(df):
      """
    Split a merged DataFrame into feature matrix X and label vector y.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'id' and 'label', and feature columns.

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Label vector of shape (n_samples,).
    """
  
    y = df["label"].values.astype(int)
    X = df.drop(columns=["id","label"]).values.astype(float)
    return X, y

def plot_curves(y_true, prob, out_prefix):
      """
    Plot ROC and precision-recall curves and save them as PNG files.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth binary labels.
    prob : numpy.ndarray
        Predicted positive class probabilities.
    out_prefix : str
        Prefix used for naming the output PNG files.
    """
    fpr, tpr, _ = roc_curve(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)

    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC")
    plt.grid(True, ls="--", alpha=0.4)
    plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_roc.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR")
    plt.grid(True, ls="--", alpha=0.4)
    plt.savefig(os.path.join(RESULTS_DIR, f"{out_prefix}_pr.png"), dpi=300)
    plt.close()

def main():
#  Main entry point for training and evaluating the stacking ensemble.
    parser = argparse.ArgumentParser()

    parser.add_argument("--sampling_strategy", type=str, default=str(SMOTE_PARAMS["sampling_strategy"]))
    parser.add_argument("--k_neighbors", type=int, default=SMOTE_PARAMS["k_neighbors"])
    parser.add_argument("--random_state", type=int, default=SMOTE_PARAMS["random_state"])
    parser.add_argument("--n_jobs", type=int, default=SMOTE_PARAMS["n_jobs"])
    args = parser.parse_args()

    train, test = prepare_data()
    X_tr, y_tr = split_xy(train)
    X_te, y_te = split_xy(test)


    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    smote = SMOTE(sampling_strategy=(args.sampling_strategy if args.sampling_strategy!="auto"
                                     else "auto"),
                  k_neighbors=args.k_neighbors,
                  random_state=args.random_state,
                  n_jobs=args.n_jobs)
    X_bal, y_bal = smote.fit_resample(X_tr_s, y_tr)
    print(f"[SMOTE] from {np.bincount(y_tr)} to {np.bincount(y_bal)}")


    rf  = RandomForestClassifier(**RF_PARAMS)
    xgb = XGBClassifier(**XGB_PARAMS)
    lgb = LGBMClassifier(**LGB_PARAMS)


    meta = LogisticRegression(**LOGREG_PARAMS)


    clf = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)],
        final_estimator=meta,
        passthrough=False,
        cv=5,
        n_jobs=-1
    )
    clf.fit(X_bal, y_bal)


    prob = clf.predict_proba(X_te_s)[:,1]
    pred = (prob >= 0.5).astype(int)

    acc  = accuracy_score(y_te, pred)
    pre  = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    f1   = f1_score(y_te, pred, zero_division=0)
    mcc  = matthews_corrcoef(y_te, pred)
    auroc= roc_auc_score(y_te, prob)
    aupr = average_precision_score(y_te, prob)

    print("\n=== Test Metrics ===")
    print(f"Accuracy: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    print(f"AUROC:   {auroc:.4f} | AUPRC:    {aupr:.4f}")
    print("\nClassification report:\n", classification_report(y_te, pred, digits=4))

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([dict(Accuracy=acc, Precision=pre, Recall=rec, F1=f1, MCC=mcc,
                       AUROC=auroc, AUPRC=aupr)]).to_csv(
        os.path.join(RESULTS_DIR, "metrics.csv"), index=False
    )
    plot_curves(y_te, prob, out_prefix="stacking")
    print(f"[OK] metrics & curves -> {RESULTS_DIR}")

    tr_out = train.copy()
    tr_out.to_csv(os.path.join(RESULTS_DIR, "train_merged_original.csv"), index=False)
    pd.DataFrame(np.hstack([X_bal, y_bal.reshape(-1,1)])).to_csv(
        os.path.join(RESULTS_DIR, "train_balanced_array.csv"), index=False, header=False
    )

if __name__ == "__main__":
    main()

