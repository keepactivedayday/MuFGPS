# [ADDED] """
# [ADDED] struct_and_contact.py
# [ADDED]
# [ADDED] This script extracts secondary structure statistics using DSSP and
# [ADDED] residue-level Cα–Cα contact maps from PDB structures. The DSSP-derived
# [ADDED] features are saved to SEC_FEAT_CSV, and the contact maps are saved
# [ADDED] as compressed NumPy archives in CONTACT_DIR.
# [ADDED]
# [ADDED] Usage:
# [ADDED]     python struct_and_contact.py
# [ADDED]
# [ADDED] Requirements:
# [ADDED]     - PDB files in PDB_DIR, named as <id>.pdb
# [ADDED]     - A FASTA file (FASTA_FILE) for optional ID cross-checking
# [ADDED]     - BASIC_FEATURES_CSV with at least an 'id' column
# [ADDED] """
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, DSSP
from Bio import SeqIO
from config import (PDB_DIR, FASTA_FILE, BASIC_FEATURES_CSV,
                    SEC_FEAT_CSV, CONTACT_DIR, CA_DIST_THRESHOLD,
                    MAX_SEQ_LEN_WARN)
from config import DSSP_BIN

warnings.filterwarnings("ignore", category=FutureWarning)

def normalize_id(x: str) -> str:

    b = os.path.basename(str(x)).split()[0]
    return os.path.splitext(b)[0]

def dssp_secondary(pdb_path: str):
    """
    Compute DSSP-based secondary structure statistics for a PDB structure.

    The function runs DSSP on the first model of the structure and counts:
        - helix residues (H, G, I)
        - sheet residues (E, B)
        - turn residues (T)

    Fractions are computed relative to the total number of residues with DSSP
    assignments.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.

    Returns
    -------
    dict
        Dictionary containing:
            - n_dssp: number of residues with DSSP records
            - helix_cnt, sheet_cnt, turn_cnt
            - helix_frac, sheet_frac, turn_frac
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = list(structure)[0]

    # Use external DSSP binary if provided, otherwise Bio.PDB's default
    if DSSP_BIN:
        dssp = DSSP(model, pdb_path, dssp=DSSP_BIN)
    else:
        dssp = DSSP(model, pdb_path)

    codes = [rec[2] for rec in dssp]
    n = len(codes)

    helix_cnt = sum(c in {"H", "G", "I"} for c in codes)
    sheet_cnt = sum(c in {"E", "B"} for c in codes)
    turn_cnt  = sum(c == "T" for c in codes)

    out = dict(
        n_dssp = n,
        helix_cnt = helix_cnt,
        sheet_cnt = sheet_cnt,
        turn_cnt  = turn_cnt,
        helix_frac = (helix_cnt / n) if n else 0.0,
        sheet_frac = (sheet_cnt / n) if n else 0.0,
        turn_frac  = (turn_cnt  / n) if n else 0.0,
    )
    return out

def ca_contact_map(pdb_path: str, thr=8.0) -> np.ndarray:
    """
    Compute a binary Cα–Cα contact map from a PDB structure.

    Contacts are defined between pairs of Cα atoms whose Euclidean distance
    is less than the given threshold (in Ångström). The diagonal is set to 0.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.
    thr : float, optional
        Distance threshold (in Å) for defining contacts.

    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix of shape (N, N), where N is the number of
        residues with Cα atoms.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = list(structure)[0]

  # Collect Cα coordinates for all residues in the model
    cas = []
    for chain in model:
        for res in chain:
            if "CA" in res:
                cas.append(res["CA"].get_coord())
    if len(cas) == 0:
        raise ValueError(f"No CA atoms in {pdb_path}")
    coords = np.vstack(cas)  # [N,3]
    if coords.shape[0] > MAX_SEQ_LEN_WARN:
        print(f"[Warn] very long protein ({coords.shape[0]} residues): {pdb_path}")

   # Compute pairwise squared distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
  
   # Build adjacency matrix based on distance threshold
    adj = (dist2 < (thr * thr)).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    return adj

def main():
    df_ids = pd.read_csv(BASIC_FEATURES_CSV)
    assert "id" in df_ids.columns, "basic_features.csv 需包含列: id"
    ids = [normalize_id(x) for x in df_ids["id"].tolist()]


    fasta_ids = set(normalize_id(r.id) for r in SeqIO.parse(FASTA_FILE, "fasta"))

    rows = []
    for pid in tqdm(ids, desc="DSSP + Contact"):
        pdb_path = os.path.join(PDB_DIR, f"{pid}.pdb")
        if not os.path.isfile(pdb_path):
            print(f"[Skip] missing PDB: {pid}")
            continue
        if pid not in fasta_ids:
            print(f"[Warn] {pid} not found in FASTA; continue using PDB only.")


        try:
            sec = dssp_secondary(pdb_path)
            rows.append(dict(id=pid, **sec))
        except Exception as e:
            print(f"[DSSP fail] {pid}: {e}")
            rows.append(dict(id=pid, n_dssp=0,
                             helix_cnt=0, sheet_cnt=0, turn_cnt=0,
                             helix_frac=0.0, sheet_frac=0.0, turn_frac=0.0))

        try:
            adj = ca_contact_map(pdb_path, thr=CA_DIST_THRESHOLD)
            np.savez_compressed(os.path.join(CONTACT_DIR, f"{pid}.npz"),
                                adj=adj.astype(np.uint8))
        except Exception as e:
            print(f"[Contact fail] {pid}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(SEC_FEAT_CSV, index=False)
        print(f"[OK] Secondary structure saved -> {SEC_FEAT_CSV}")
    else:
        print("[WARN] No secondary structure rows produced.")

if __name__ == "__main__":
    main()

