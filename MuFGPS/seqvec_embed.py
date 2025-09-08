
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
