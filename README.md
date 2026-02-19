MuFGPS is a modular and reproducible machine learning pipeline for predicting liquid-liquid phase separation (LLPS) proteins.  It integrates sequence features, secondary structure features, and graph-based structural features extracted from protein contact maps using a graph attention network (GAT).  Final classification is performed using a stacking ensemble of tree-based models.

1.Environment

Recommended Python version: ≥ 3.7

Install dependencies using:

pip install -r requirements.txt

If DSSP is required, install it separately and specify the path in config.py (DSSP_BIN).


2.Data Preparation

Place the following files in the data/ directory:

basic_features.csv (must contain columns: id, label)

data.fasta (protein sequences)

protein_pdb/ (directory containing PDB files named as <id>.pdb)

All paths and hyperparameters are defined in config.py.


3.Reproducibility Workflow

Run the following scripts in order:

Step 1 – Generate SeqVec embeddings:
python seqvec_embed.py

Step 2 – Extract secondary structure features and contact maps:
python struct_and_contact.py

Step 3 – Train GAT and extract graph embeddings:
python gat.py

Step 4 – Train stacking ensemble classifier:
python ensemble.py

4.Outputs

All outputs are stored in the outputs/ directory, including:

secondary_features.csv

contact_map/ (contact matrices)

embeddings/ (SeqVec embeddings)

gat_features.csv

results/metrics.csv

ROC and PR curves

5.Reproducibility Settings

Fixed random seed (RANDOM_SEED in config.py)

Stratified train/test split

SMOTE applied only to training data

All hyperparameters defined in config.py
