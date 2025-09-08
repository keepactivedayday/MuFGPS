
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
from config import (
    BASIC_FEATURES_CSV, CONTACT_DIR, EMBED_DIR, SPLIT_CSV, GAT_CKPT, GAT_FEAT_CSV,
    EMBED_DIM, GAT_HIDDEN, GAT_OUT_DIM, HEADS, DROPOUT, LEAKY_NEG_SLOPE,
    EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE,
    RANDOM_SEED, TEST_SIZE
)

torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def norm_id(x: str) -> str:
    return os.path.basename(str(x)).split()[0].split("|")[0]

def load_graph(pid: str):


    cpath = os.path.join(CONTACT_DIR, f"{pid}.npz")
    epath = os.path.join(EMBED_DIR, f"{pid}.npz")
    if not (os.path.isfile(cpath) and os.path.isfile(epath)):
        return None

    adj = np.load(cpath)["adj"].astype(np.uint8)
    x = np.load(epath)["x"].astype(np.float32)

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
    def __init__(self):
        super().__init__()
        self.g1 = GATv2Conv(EMBED_DIM, GAT_HIDDEN, heads=HEADS[0],
                            dropout=DROPOUT, edge_dim=None)
        self.g2 = GATv2Conv(GAT_HIDDEN * HEADS[0], GAT_HIDDEN, heads=HEADS[1],
                            dropout=DROPOUT)

        self.g3 = GATv2Conv(GAT_HIDDEN * HEADS[1], GAT_OUT_DIM, heads=HEADS[2],
                            concat=False, dropout=DROPOUT)
        self.act = nn.LeakyReLU(LEAKY_NEG_SLOPE)
        self.cls = nn.Linear(GAT_OUT_DIM, 2)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, batch):
        x = self.act(self.g1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.g2(x, edge_index))
        x = self.dropout(x)
        x = self.g3(x, edge_index)  # [N, 128]
        g = global_max_pool(x, batch)  # [B, 128]
        logits = self.cls(g)
        return logits, g

def get_split(df):
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

def train_one_epoch(model, loader, optim, criterion):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optim.step()
        total += float(loss) * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def eval_loss(model, loader, criterion):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y.view(-1))
        total += float(loss) * batch.num_graphs
    return total / len(loader.dataset)

@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    feats = []
    pids = []
    ys = []
    for batch in loader:
        batch = batch.to(device)
        _, g = model(batch.x, batch.edge_index, batch.batch)
        feats.append(g.cpu().numpy())
        pids.extend(batch.pid)
        ys.extend(batch.y.view(-1).cpu().numpy())
    X = np.vstack(feats)
    return pids, ys, X

def main():
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
    criterion = nn.CrossEntropyLoss()

    best_loss = 1e9
    patience = PATIENCE
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, optim, criterion)
        te_loss = eval_loss(model, test_loader, criterion)
        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | test {te_loss:.4f}")
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
