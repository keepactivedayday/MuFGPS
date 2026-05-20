MuFGPS is a modular and reproducible machine learning pipeline for predicting liquid-liquid phase separation (LLPS) proteins.  It integrates sequence features, secondary structure features, and graph-based structural features extracted from protein contact maps using a graph attention network (GAT).  Final classification is performed using a stacking ensemble of tree-based models.


Recommended Python version: ≥ 3.7

Install dependencies using:

pip install -r requirements.txt

If DSSP is required, install it separately and specify the path in config.py (DSSP_BIN).

