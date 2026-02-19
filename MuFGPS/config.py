"""
config.py

Central configuration file for the MuFGPS framework.

This file defines:
    - Project directories
    - Data input paths
    - Output paths
    - Structural parameters
    - GAT hyperparameters
    - Training parameters
    - SMOTE settings
    - Ensemble classifier parameters

All scripts import configuration variables from this file to ensure
experimental consistency and reproducibility.
"""

import os


# ==========================================================
# Project root
# ==========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ==========================================================
# Data paths
# ==========================================================

DATA_DIR = os.path.join(BASE_DIR, "data")

BASIC_FEATURES_CSV = os.path.join(DATA_DIR, "basic_features.csv")
FASTA_FILE = os.path.join(DATA_DIR, "data.fasta")
PDB_DIR = os.path.join(DATA_DIR, "protein_pdb")

# Optional external DSSP executable (leave "" to use default)
DSSP_BIN = ""


# ==========================================================
# Output paths
# ==========================================================

OUT_DIR = os.path.join(BASE_DIR, "outputs")

SEC_FEAT_CSV = os.path.join(OUT_DIR, "secondary_features.csv")
CONTACT_DIR = os.path.join(OUT_DIR, "contact_map")
EMBED_DIR = os.path.join(OUT_DIR, "embeddings")
SPLIT_CSV = os.path.join(OUT_DIR, "split.csv")

GAT_CKPT = os.path.join(OUT_DIR, "checkpoints", "gat.pt")
GAT_FEAT_CSV = os.path.join(OUT_DIR, "gat_features.csv")

MERGED_TRAIN_CSV = os.path.join(OUT_DIR, "train_merged.csv")
MERGED_TEST_CSV  = os.path.join(OUT_DIR, "test_merged.csv")

RESULTS_DIR = os.path.join(OUT_DIR, "results")


# Automatically create required directories
for d in [
    OUT_DIR,
    CONTACT_DIR,
    EMBED_DIR,
    os.path.join(OUT_DIR, "checkpoints"),
    RESULTS_DIR
]:
    os.makedirs(d, exist_ok=True)


# ==========================================================
# Structural feature parameters
# ==========================================================

# Cα–Cα distance threshold (Å) for defining residue contacts
CA_DIST_THRESHOLD = 8.0

# Warn if protein sequence exceeds this length
MAX_SEQ_LEN_WARN = 6000


# ==========================================================
# Reproducibility & data split
# ==========================================================

RANDOM_SEED = 42
TEST_SIZE = 0.2


# ==========================================================
# GAT architecture parameters
# ==========================================================

EMBED_DIM = 1024          # Dimension of residue-level embeddings
GAT_HIDDEN = 128          # Hidden layer size
HEADS = [3, 3, 1]         # Attention heads per layer
GAT_OUT_DIM = 128         # Graph-level embedding dimension

DROPOUT = 0.2
LEAKY_NEG_SLOPE = 0.2


# ==========================================================
# GAT training hyperparameters
# ==========================================================

EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6


# ==========================================================
# SMOTE parameters
# ==========================================================

SMOTE_PARAMS = dict(
    sampling_strategy="auto",
    k_neighbors=5,
    n_jobs=1,
    random_state=RANDOM_SEED
)


# ==========================================================
# Base learners (Stacking ensemble)
# ==========================================================

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    eval_metric="logloss"
)

LGB_PARAMS = dict(
    n_estimators=200,
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.05,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=RANDOM_SEED
)


# ==========================================================
# Meta-learner (Logistic Regression)
# ==========================================================

LOGREG_PARAMS = dict(
    C=1.0,
    solver="lbfgs",
    max_iter=500,
    random_state=RANDOM_SEED
)
