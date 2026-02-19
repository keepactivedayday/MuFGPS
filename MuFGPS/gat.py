# [ADDED] """
# [ADDED] gat.py
# [ADDED]
# [ADDED] This script trains a Graph Attention Network (GAT) on residue-level
# [ADDED] contact graphs derived from AlphaFold models, using a self-supervised
# [ADDED] contrastive learning objective. The trained GAT encoder is then used
# [ADDED] to extract graph-level structural embeddings for each protein, which
# [ADDED] are exported to a CSV file (GAT_FEAT_CSV) and later used as structural
# [ADDED] features in the MuFGPS framework.
# [ADDED]
# [ADDED] Usage:
# [ADDED]     python gat.py
# [ADDED]
# [ADDED] Requirements:
# [ADDED]     - Precomputed residue contact matrices in CONTACT_DIR (one .npz per protein).
# [ADDED]     - Precomputed per-residue embeddings in EMBED_DIR (e.g., language model embeddings).
# [ADDED]     - BASIC_FEATURES_CSV containing at least columns ['id', 'label'].
# [ADDED] """
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_max_pool
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F 

from config import (
    BASIC_FEATURES_CSV, CONTACT_DIR, EMBED_DIR, SPLIT_CSV, GAT_CKPT, GAT_FEAT_CSV,
    EMBED_DIM, GAT_HIDDEN, GAT_OUT_DIM, HEADS, DROPOUT, LEAKY_NEG_SLOPE,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE,
    RANDOM_SEED, TEST_SIZE
)

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_id(x: str) -> str:
        """
    Normalize protein identifiers.

    This function extracts a clean protein ID from various possible ID formats
    (e.g., stripping path, whitespace, or '|' separated fields).

    Parameters
    ----------
    x : str
        Raw identifier string.

    Returns
    -------
    str
        Normalized protein ID.
    """
    return os.path.basename(str(x)).split()[0].split("|")[0]

def load_graph(pid: str):
        """
    Load a residue-level contact graph and corresponding node features for a protein.

    Parameters
    ----------
    pid : str
        Normalized protein ID.

    Returns
    -------
    torch_geometric.data.Data or None
        A PyG Data object containing:
            - x: node feature matrix (num_nodes, EMBED_DIM)
            - edge_index: undirected edge index based on the contact map
        Returns None if either contact or embedding file is missing.
    """
    cpath = os.path.join(CONTACT_DIR, f"{pid}.npz")
    epath = os.path.join(EMBED_DIR, f"{pid}.npz")
    if not (os.path.isfile(cpath) and os.path.isfile(epath)):
        return None

# Load adjacency matrix (contact map) and per-residue embeddings
    adj = np.load(cpath)["adj"].astype(np.uint8)
    x = np.load(epath)["x"].astype(np.float32)

# Ensure that the adjacency matrix and the features are aligned in terms of the number of nodes.
    n = min(adj.shape[0], x.shape[0])
    if n < adj.shape[0]:
        adj = adj[:n, :n]
    if n < x.shape[0]:
        x = x[:n, :]

    src, dst = np.nonzero(adj)
    edges = np.vstack([np.hstack([src, dst]), np.hstack([dst, src])])
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges).long())
    data.num_nodes = n
    return data

class GraphSet(Dataset):
    """
    Dataset wrapper for a collection of protein graphs.

    Each item corresponds to one protein graph with:
        - residue-level node features
        - residue contact edges
        - graph-level label (LLPS vs non-LLPS)

    Graphs that cannot be constructed (e.g., missing contact/embedding files)
    are skipped.
    """
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
        self.items = []
        for pid, y in tqdm(list(zip(ids, labels)), desc="Load graphs"):
            g = load_graph(pid)
            if g is not None:
                g.y = torch.tensor([y], dtype=torch.long)
                g.pid = pid
                self.items.append(g)
        if len(self.items) == 0:
            raise RuntimeError("No graphs loaded. Check CONTACT_DIR / EMBED_DIR.")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class GATNet(nn.Module):
    """
    Graph Attention Network encoder for protein contact graphs.

    Architecture:
        - Three GATv2Conv layers with multi-head attention.
        - Global max pooling to obtain graph-level representation.
        - A small projection MLP on top of the pooled embedding for
          contrastive learning.

    The model returns:
        - g: pooled graph-level embedding
        - z: projected embedding used in the contrastive objective
    """
    def __init__(self):
        super().__init__()
        self.g1 = GATv2Conv(EMBED_DIM, GAT_HIDDEN, heads=HEADS[0],
                            dropout=DROPOUT, edge_dim=None)
        self.g2 = GATv2Conv(GAT_HIDDEN * HEADS[0], GAT_HIDDEN, heads=HEADS[1],
                            dropout=DROPOUT)

        self.g3 = GATv2Conv(GAT_HIDDEN * HEADS[1], GAT_OUT_DIM, heads=HEADS[2],
                            concat=False, dropout=DROPOUT)
        self.act = nn.LeakyReLU(LEAKY_NEG_SLOPE)
        self.dropout = nn.Dropout(DROPOUT)

        self.proj = nn.Sequential(
            nn.Linear(GAT_OUT_DIM, GAT_OUT_DIM),
            nn.ReLU(),
            nn.Linear(GAT_OUT_DIM, GAT_OUT_DIM)
        )


    def forward(self, x, edge_index, batch):
            """
        Forward pass of the GAT encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (num_nodes, EMBED_DIM).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        batch : torch.Tensor
            Batch vector that maps each node to its graph index.

        Returns
        -------
        g : torch.Tensor
            Graph-level embedding after global pooling.
        z : torch.Tensor
            Projected embedding used for contrastive learning.
        """
        x = self.act(self.g1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.g2(x, edge_index))
        x = self.dropout(x)
        x = self.g3(x, edge_index) 
        g = global_max_pool(x, batch)  

        z = self.proj(g)  
        return g, z


def augment_graph(x, edge_index, edge_drop_p=0.2, node_mask_p=0.1):
    """
    Perform simple graph augmentations for contrastive learning.

    Two types of augmentations are applied:
        1. Edge dropout: randomly drop a subset of edges.
        2. Node feature masking: randomly mask (zero-out) a subset of node features.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix.
    edge_index : torch.Tensor
        Edge index of the graph.
    edge_drop_p : float, optional
        Probability of dropping each edge.
    node_mask_p : float, optional
        Probability of masking each node.

    Returns
    -------
    x_aug : torch.Tensor
        Augmented node feature matrix.
    edge_index_aug : torch.Tensor
        Augmented edge index.
    """

    # Edge dropout   
    if edge_drop_p > 0.0 and edge_index.size(1) > 0:
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > edge_drop_p
        edge_index = edge_index[:, keep_mask]

     # Node feature masking
    if node_mask_p > 0.0 and x.size(0) > 0:
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes, device=x.device) < node_mask_p
        x = x.clone()
        x[mask] = 0.0

    return x, edge_index


def contrastive_loss(z1, z2, temperature=0.2):
    """
    Compute a simple contrastive loss between two augmented views.

    The embeddings from two augmented views (z1, z2) of the same batch are
    treated as positive pairs, while all other pairs within the batch are
    treated as negatives. A temperature-scaled cosine similarity matrix is
    used as logits for a cross-entropy loss.

    Parameters
    ----------
    z1 : torch.Tensor
        Projected embeddings from the first view, shape (N, d).
    z2 : torch.Tensor
        Projected embeddings from the second view, shape (N, d).
    temperature : float, optional
        Temperature scaling factor.

    Returns
    -------
    torch.Tensor
        Scalar contrastive loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  

    sim = torch.matmul(z, z.t()) / temperature 
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)  

   
    pos_idx = torch.arange(N, 2 * N, device=z.device)
    labels = torch.cat([pos_idx, torch.arange(0, N, device=z.device)], dim=0)
    loss = F.cross_entropy(sim, labels)
    return loss


def train_one_epoch(model, loader, optim):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()

      
        x1, e1 = augment_graph(batch.x, batch.edge_index)
        x2, e2 = augment_graph(batch.x, batch.edge_index)

        _, z1 = model(x1, e1, batch.batch)
        _, z2 = model(x2, e2, batch.batch)

        loss = contrastive_loss(z1, z2)
        loss.backward()
        optim.step()
        total += float(loss) * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)

        x1, e1 = augment_graph(batch.x, batch.edge_index)
        x2, e2 = augment_graph(batch.x, batch.edge_index)

        _, z1 = model(x1, e1, batch.batch)
        _, z2 = model(x2, e2, batch.batch)

        loss = contrastive_loss(z1, z2)
        total += float(loss) * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def extract_features(model, loader):
    """
    Extract graph-level embeddings for all proteins using the trained GAT encoder.

    Parameters
    ----------
    model : nn.Module
        Trained GATNet model (encoder).
    loader : DataLoader
        DataLoader over GraphSet containing all proteins.

    Returns
    -------
    pids : list of str
        Protein IDs corresponding to each embedding.
    ys : list of int
        Labels (LLPS vs non-LLPS) for each protein.
    X : numpy.ndarray
        Graph-level embeddings of shape (n_proteins, GAT_OUT_DIM).
    """
    model.eval()
    feats = []
    pids = []
    ys = []
    for batch in loader:
        batch = batch.to(device)
        g, _ = model(batch.x, batch.edge_index, batch.batch)  
        feats.append(g.cpu().numpy())
        pids.extend(batch.pid)
        ys.extend(batch.y.view(-1).cpu().numpy())
    X = np.vstack(feats)
    return pids, ys, X

def get_split(df):
    """
    Obtain or create a train/test split based on BASIC_FEATURES_CSV.

    If a 'split' column already exists, it is reused. Otherwise, a new
    stratified split is created and saved to SPLIT_CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'id' and 'label' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['id', 'label', 'split'].
    """
    if "split" in df.columns:
        return df[["id", "label", "split"]]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    ids = df["id"].apply(norm_id).values
    y = df["label"].values
    (train_idx, test_idx) = next(sss.split(ids, y))
    dd = pd.DataFrame(dict(id=ids, label=y))
    dd.loc[train_idx, "split"] = "train"
    dd.loc[test_idx, "split"] = "test"
    dd.to_csv(SPLIT_CSV, index=False)
    print(f"[OK] split saved -> {SPLIT_CSV}")
    return dd

def main():
"""
    Main entry point for training the GAT encoder and exporting graph embeddings.
"""
    df = pd.read_csv(BASIC_FEATURES_CSV)
    assert {"id","label"}.issubset(df.columns)
    split_df = get_split(df)
    tr = split_df[split_df["split"]=="train"]
    te = split_df[split_df["split"]=="test"]

    train_set = GraphSet(tr["id"].apply(norm_id).tolist(), tr["label"].tolist())
    test_set  = GraphSet(te["id"].apply(norm_id).tolist(), te["label"].tolist())

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    model = GATNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  
    

    best_loss = 1e9
    patience = PATIENCE
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, optim) 
        te_loss = eval_loss(model, test_loader)               
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {te_loss:.4f}")
        if te_loss < best_loss - 1e-4:
            best_loss = te_loss
            patience = PATIENCE
            torch.save(model.state_dict(), GAT_CKPT)
        else:
            patience -= 1
            if patience <= 0:
                print("[Early stop]")
                break
    print(f"[OK] best model saved -> {GAT_CKPT}")

    model.load_state_dict(torch.load(GAT_CKPT, map_location=device))

    all_ids = split_df["id"].apply(norm_id).tolist()
    all_labels = split_df["label"].tolist()
    all_set = GraphSet(all_ids, all_labels)
    all_loader = DataLoader(all_set, batch_size=BATCH_SIZE, shuffle=False)
    pids, ys, X = extract_features(model, all_loader)

    cols = [f"gat_{i:03d}" for i in range(X.shape[1])]
    out = pd.DataFrame(X, columns=cols)
    out.insert(0, "id", pids)
    out.insert(1, "label", ys)
    out = out.drop_duplicates(subset=["id"])
    out.to_csv(GAT_FEAT_CSV, index=False)
    print(f"[OK] GAT features -> {GAT_FEAT_CSV}")

if __name__ == "__main__":
    main()

