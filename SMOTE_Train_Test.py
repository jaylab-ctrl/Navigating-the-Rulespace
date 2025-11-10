import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.optim import AdamW
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "LegalroBERTa_Citations_Label.csv"
LABEL_COL = "original_label"
TEXT_CANDIDATES = ["text","citation_text","context","content","section_text","clean_text"]
ID_CANDIDATES = ["id","doc_id","row_id","uuid","hash"]

# data loading and column selection
df = pd.read_csv(DATA_PATH)
text_col = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
if text_col is None: raise ValueError(f"No text column found among {TEXT_CANDIDATES}")
id_col = next((c for c in ID_CANDIDATES if c in df.columns), None)
if id_col is None:
    df["__row_id__"] = np.arange(len(df))
    id_col = "__row_id__"
if LABEL_COL not in df.columns: raise ValueError(f"Target '{LABEL_COL}' not found")
df = df.dropna(subset=[text_col, LABEL_COL]).reset_index(drop=True)

# label encoding
labels = df[LABEL_COL].astype(str).values
classes = np.unique(labels).tolist()
class_to_id = {c:i for i,c in enumerate(classes)}
id_to_class = {i:c for c,i in class_to_id.items()}
y = np.array([class_to_id[c] for c in labels], dtype=np.int64)
texts = df[text_col].astype(str).tolist()
ids = df[id_col].astype(str).tolist()

def show_distribution(name, y_vals):
    c = Counter(y_vals); total = len(y_vals)
    dist = {id_to_class[k]: f"{v} ({v/total:.2%})" for k,v in sorted(c.items())}
    print(name, "class distribution:", dist)

# stratified split to keep target proportions equal in train/test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(sss.split(np.arange(len(y)), y))
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

y_train, y_test = y[train_idx], y[test_idx]
texts_train = [texts[i] for i in train_idx]
texts_test  = [texts[i] for i in test_idx]
ids_train   = [ids[i] for i in train_idx]
ids_test    = [ids[i] for i in test_idx]

show_distribution("TRAIN", y_train)
show_distribution("TEST", y_test)

Path("./artifacts").mkdir(parents=True, exist_ok=True)
Path("./artifacts/test_ids.json").write_text(json.dumps(ids_test, indent=2))

# helper to extract frozen transformer embeddings
def embed_texts(model_name, texts, batch_size=16, max_length=384, pool="cls"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k:v.to(DEVICE) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            if hasattr(out, "last_hidden_state"):
                if pool == "cls":
                    e = out.last_hidden_state[:,0,:]
                else:
                    e = out.last_hidden_state.mean(dim=1)
            else:
                e = out[0][:,0,:]
        embs.append(e.detach().cpu().numpy())
    E = np.vstack(embs).astype(np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    return E / norms

# build classical TF-IDF+SVD feature space for GCN baseline"
tfidf = TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=100000)
Xtr_tfidf = tfidf.fit_transform(texts_train)
svd = TruncatedSVD(n_components=300, random_state=SEED)
Xtr_svd = svd.fit_transform(Xtr_tfidf)
scaler = StandardScaler(with_mean=True, with_std=True)
Xtr_svd = scaler.fit_transform(Xtr_svd)
Xte_svd = scaler.transform(svd.transform(tfidf.transform(texts_test)))

# SMOTE only within training folds and class-weighted LogisticRegression on SVD features
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores_svd = []
for tr, va in skf.split(Xtr_svd, y_train):
    X_tr, y_tr = Xtr_svd[tr], y_train[tr]
    X_va, y_va = Xtr_svd[va], y_train[va]
    sm = SMOTE(k_neighbors=5, random_state=SEED)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr_sm, y_tr_sm)
    y_hat = clf.predict(X_va)
    cv_scores_svd.append(f1_score(y_va, y_hat, average="macro", zero_division=0))

print("CV macro-F1 (SVD+LogReg with in-fold SMOTE):", round(float(np.mean(cv_scores_svd)),4))

sm = SMOTE(k_neighbors=5, random_state=SEED)
Xtr_svd_sm, ytr_svd_sm = sm.fit_resample(Xtr_svd, y_train)
svd_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
svd_clf.fit(Xtr_svd_sm, ytr_svd_sm)
svd_pred = svd_clf.predict(Xte_svd)

print("SVD+LogReg TEST report")
print(classification_report(y_test, svd_pred, target_names=[id_to_class[i] for i in range(len(classes))], digits=4))

# frozen LegalBERT embeddings + SMOTE-in-folds + class-weighted LogisticRegression"
Etr_legal = embed_texts("nlpaueb/legal-bert-base-uncased", texts_train)
Ete_legal = embed_texts("nlpaueb/legal-bert-base-uncased", texts_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores_legal = []

for tr, va in skf.split(Etr_legal, y_train):
    X_tr, y_tr = Etr_legal[tr], y_train[tr]
    X_va, y_va = Etr_legal[va], y_train[va]
    sm = SMOTE(k_neighbors=5, random_state=SEED)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr_sm, y_tr_sm)
    y_hat = clf.predict(X_va)
    cv_scores_legal.append(f1_score(y_va, y_hat, average="macro", zero_division=0))

print("CV macro-F1 (LegalBERT-emb + LogReg with in-fold SMOTE):", round(float(np.mean(cv_scores_legal)),4))

sm = SMOTE(k_neighbors=5, random_state=SEED)
Etr_legal_sm, ytr_legal_sm = sm.fit_resample(Etr_legal, y_train)

legal_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
legal_clf.fit(Etr_legal_sm, ytr_legal_sm)
legal_pred = legal_clf.predict(Ete_legal)

print("LegalBERT-emb TEST report")
print(classification_report(y_test, legal_pred, target_names=[id_to_class[i] for i in range(len(classes))], digits=4))

# frozen DistilBERT embeddings + SMOTE-in-folds + class-weighted LogisticRegression
Etr_distil = embed_texts("distilbert-base-uncased", texts_train)
Ete_distil = embed_texts("distilbert-base-uncased", texts_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores_distil = []

for tr, va in skf.split(Etr_distil, y_train):
    X_tr, y_tr = Etr_distil[tr], y_train[tr]
    X_va, y_va = Etr_distil[va], y_train[va]
    sm = SMOTE(k_neighbors=5, random_state=SEED)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    clf.fit(X_tr_sm, y_tr_sm)
    y_hat = clf.predict(X_va)
    cv_scores_distil.append(f1_score(y_va, y_hat, average="macro", zero_division=0))

print("CV macro-F1 (DistilBERT-emb + LogReg with in-fold SMOTE):", round(float(np.mean(cv_scores_distil)),4))

sm = SMOTE(k_neighbors=5, random_state=SEED)
Etr_distil_sm, ytr_distil_sm = sm.fit_resample(Etr_distil, y_train)

distil_clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
distil_clf.fit(Etr_distil_sm, ytr_distil_sm)

distil_pred = distil_clf.predict(Ete_distil)

print("DistilBERT-emb TEST report")
print(classification_report(y_test, distil_pred, target_names=[id_to_class[i] for i in range(len(classes))], digits=4))

# GCN using SVD features, class-weighted loss, no SMOTE
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X, A_hat):
        H = torch.relu(self.lin1(A_hat @ X))
        H = self.dropout(H)
        Z = self.lin2(A_hat @ H)
        return Z

def normalize_adj(A):
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d>0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

nbrs = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Xtr_svd)

distances, indices = nbrs.kneighbors(Xtr_svd)
rows, cols, vals = [], [], []
for i in range(len(Xtr_svd)):
    for j, d in zip(indices[i], distances[i]):
        if i == j: continue
        w = 1.0 - d
        rows.append(i); cols.append(j); vals.append(w)
A_train = np.zeros((len(Xtr_svd), len(Xtr_svd)), dtype=np.float32)
if len(rows): A_train[rows, cols] = vals
A_train = normalize_adj(A_train).astype(np.float32)

nbrs_te = NearestNeighbors(n_neighbors=10, metric="cosine").fit(Xtr_svd)
d_te, i_te = nbrs_te.kneighbors(Xte_svd)
rows_te, cols_te, vals_te = [], [], []
for i in range(len(Xte_svd)):
    for j, d in zip(i_te[i], d_te[i]):
        w = 1.0 - d
        rows_te.append(i); cols_te.append(j); vals_te.append(w)

A_te_tr = np.zeros((len(Xte_svd), len(Xtr_svd)), dtype=np.float32)
if len(rows_te): A_te_tr[rows_te, cols_te] = vals_te
D_train = np.diag(np.maximum(A_train.sum(1), 1e-6)).astype(np.float32)
Dte = np.diag(np.maximum(A_te_tr.sum(1), 1e-6)).astype(np.float32)
A_te_tr_norm = np.linalg.solve(np.sqrt(Dte), A_te_tr) @ np.linalg.solve(np.sqrt(D_train), np.eye(len(Xtr_svd), dtype=np.float32))

Xtr_t = torch.tensor(Xtr_svd, dtype=torch.float32, device=DEVICE)
Xte_t = torch.tensor(Xte_svd, dtype=torch.float32, device=DEVICE)
A_tr_t = torch.tensor(A_train, dtype=torch.float32, device=DEVICE)
A_te_tr_t = torch.tensor(A_te_tr_norm, dtype=torch.float32, device=DEVICE)
ytr_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
yte_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

gcn = SimpleGCN(in_dim=Xtr_svd.shape[1], hid_dim=256, out_dim=len(classes), dropout=0.2).to(DEVICE)
counts_tr = np.bincount(y_train, minlength=len(classes))
w_tr = torch.tensor((len(y_train) / (len(classes) * np.maximum(counts_tr,1))).astype(np.float32), device=DEVICE)
opt = AdamW(gcn.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=w_tr)

# GCN training (full-batch)
gcn.train()
for epoch in range(50):
    opt.zero_grad()
    logits_tr = gcn(Xtr_t, A_tr_t)
    loss = criterion(logits_tr, ytr_t)
    loss.backward()
    opt.step()

# GCN inference on TEST via train-neighbor aggregation
gcn.eval()
with torch.no_grad():
    H1_tr = torch.relu(gcn.lin1(A_tr_t @ Xtr_t))
    H1_tr = gcn.dropout(H1_tr)
    Z_te = gcn.lin2(A_te_tr_t @ H1_tr)
    gcn_pred = torch.argmax(Z_te, dim=1).cpu().numpy()

print("GCN TEST report")
print(classification_report(y_test, gcn_pred, target_names=[id_to_class[i] for i in range(len(classes))], digits=4))

# consolidated comparison
def macro_f1(y_true, y_pred): return f1_score(y_true, y_pred, average="macro", zero_division=0)
summary = {
    # Accuracy
    "Accuracy_LegalBERT": round(accuracy_score(y_test, legal_pred),4), # 95%
    "Accuracy_DistilBERT": round(accuracy_score(y_test, distil_pred),4), # 95%
    "Accuracy_GCN": round(accuracy_score(y_test, gcn_pred),4), # 84%

    # Macro-F1
    "MacroF1_LegalBERT": round(macro_f1(y_test, legal_pred),4), # 92%
    "MacroF1_DistilBERT": round(macro_f1(y_test, distil_pred),4), # 92%
    "MacroF1_GCN": round(macro_f1(y_test, gcn_pred),4), # 79%
}
print(summary)