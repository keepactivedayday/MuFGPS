# [ADDED] """
# [ADDED] seqvec_embed.py
# [ADDED]
# [ADDED] This script generates per-residue SeqVec embeddings for all proteins
# [ADDED] listed in BASIC_FEATURES_CSV and saves them as compressed .npz files
# [ADDED] in EMBED_DIR. Each .npz file contains an array `x` with shape
# [ADDED] (L, D), where L is sequence length and D is the embedding dimension.
# [ADDED]
# [ADDED] Usage:
# [ADDED]     python seqvec_embed.py
# [ADDED]
# [ADDED] Requirements:
# [ADDED]     - A FASTA file containing protein sequences (FASTA_FILE).
# [ADDED]     - BASIC_FEATURES_CSV with at least the column 'id'.
# [ADDED]     - The bio_embeddings library with SeqVecEmbedder available.
# [ADDED] """
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from bio_embeddings.embed import SeqVecEmbedder
from config import (FASTA_FILE, BASIC_FEATURES_CSV, EMBED_DIR)

def normalize_id(x: str) -> str:
    return os.path.basename(str(x)).split()[0].split("|")[0]

def main():
    """
    Main entry point for generating SeqVec embeddings.

    Workflow:
        1. Read BASIC_FEATURES_CSV and collect the set of protein IDs of interest.
        2. Iterate over sequences in FASTA_FILE.
        3. For each sequence whose ID appears in BASIC_FEATURES_CSV, generate
           a SeqVec embedding (after replacing uncommon amino acids) and save
           it to EMBED_DIR as a compressed .npz file.
        4. Existing embedding files are skipped to avoid redundant computation.
    """
    df = pd.read_csv(BASIC_FEATURES_CSV)
    assert "id" in df.columns and "label" in df.columns, "basic_features.csv 需含 id,label"
    wanted = set(normalize_id(x) for x in df["id"].tolist())


    embedder = SeqVecEmbedder()

    for rec in tqdm(SeqIO.parse(FASTA_FILE, "fasta"), desc="SeqVec"):
        pid = normalize_id(rec.id)
        if pid not in wanted:
            continue
        out_path = os.path.join(EMBED_DIR, f"{pid}.npz")
        if os.path.exists(out_path):
            continue
        seq = str(rec.seq).replace("U", "X").replace("O", "X")
        emb = embedder.embed(seq)
        np.savez_compressed(out_path, x=emb.astype(np.float32))
    print("[OK] SeqVec embeddings saved.")

if __name__ == "__main__":
    main()

