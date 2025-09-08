
import os


DATA_DIR = "data"
BASIC_FEATURES_CSV = os.path.join(DATA_DIR, "basic_features.csv")
FASTA_FILE = os.path.join(DATA_DIR, "data.fasta")
PDB_DIR = os.path.join(DATA_DIR, "protein_pdb")
DSSP_BIN = ""


OUT_DIR = "outputs"
SEC_FEAT_CSV = os.path.join(OUT_DIR, "secondary_features.csv")
CONTACT_DIR = os.path.join(OUT_DIR, "contact_map")
EMBED_DIR = os.path.join(OUT_DIR, "embeddings")
SPLIT_CSV = os.path.join(OUT_DIR, "split.csv")
GAT_CKPT = os.path.join(OUT_DIR, "checkpoints", "gat.pt")
GAT_FEAT_CSV = os.path.join(OUT_DIR, "gat_features.csv")
MERGED_TRAIN_CSV = os.path.join(OUT_DIR, "train_merged.csv")
MERGED_TEST_CSV  = os.path.join(OUT_DIR, "test_merged.csv")
RESULTS_DIR = os.path.join(OUT_DIR, "results")


for d in [OUT_DIR, CONTACT_DIR, EMBED_DIR, os.path.join(OUT_DIR, "checkpoints"), RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


CA_DIST_THRESHOLD = 8.0
MAX_SEQ_LEN_WARN = 6000


RANDOM_SEED = 42
TEST_SIZE = 0.2


EMBED_DIM = 1024
GAT_HIDDEN = 128
HEADS = [3, 3, 1]
GAT_OUT_DIM = 128
DROPOUT = 0.2
LEAKY_NEG_SLOPE = 0.2


EPOCHS = 40
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6


SMOTE_PARAMS = dict(
    sampling_strategy="auto",
    k_neighbors=5,
    n_jobs=1,
    random_state=RANDOM_SEED
)


RF_PARAMS = dict(n_estimators=200, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt",
                 random_state=RANDOM_SEED, n_jobs=-1)

XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
                  random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss")

LGB_PARAMS = dict(n_estimators=200, num_leaves=31, max_depth=-1, learning_rate=0.05,
                  min_child_samples=10, subsample=0.9, colsample_bytree=0.8,
                  reg_alpha=0.0, reg_lambda=0.0, random_state=RANDOM_SEED)


LOGREG_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=500, random_state=RANDOM_SEED)
