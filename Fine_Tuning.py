import os, json, random, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH="LegalroBERTa_Citations_Label.csv"
LABEL_COL="original_label"
TEXT_CANDIDATES=["text","citation_text","context","content","section_text","clean_text"]
ID_CANDIDATES=["id","doc_id","row_id","uuid","hash"]

#data loading and column selection
df=pd.read_csv(DATA_PATH)

text_col=next((c for c in TEXT_CANDIDATES if c in df.columns),None)

if text_col is None: raise ValueError(f"No text column among {TEXT_CANDIDATES}")
id_col=next((c for c in ID_CANDIDATES if c in df.columns),None)

if id_col is None:
    df["__row_id__"]=np.arange(len(df)); id_col="__row_id__"

if LABEL_COL not in df.columns: raise ValueError(f"Target '{LABEL_COL}' not found")
df=df.dropna(subset=[text_col,LABEL_COL]).reset_index(drop=True)

# label encoding and stratified split (equal proportions in train/test)"
labels=df[LABEL_COL].astype(str).values

classes=sorted(np.unique(labels).tolist())
class_to_id={c:i for i,c in enumerate(classes)}
id_to_class={i:c for c,i in class_to_id.items()}

y=np.array([class_to_id[c] for c in labels],dtype=np.int64)

texts=df[text_col].astype(str).tolist()

ids=df[id_col].astype(str).tolist()

sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)
train_idx,test_idx=next(sss.split(np.arange(len(y)),y))
train_idx,test_idx=np.array(train_idx),np.array(test_idx)

y_train,y_test=y[train_idx],y[test_idx]

texts_train=[texts[i] for i in train_idx]
texts_test=[texts[i] for i in test_idx]

ids_test=[ids[i] for i in test_idx]

Path("./artifacts").mkdir(parents=True,exist_ok=True)
Path("./artifacts/test_ids.json").write_text(json.dumps(ids_test,indent=2))

# metric helpers
def compute_metrics_eval(pred):
    logits, labels = pred
    preds=np.argmax(logits,axis=1)
    acc=accuracy_score(labels,preds)
    p,r,f1,_=precision_recall_fscore_support(labels,preds,average="macro",zero_division=0)
    return {"accuracy":acc,"macro_precision":p,"macro_recall":r,"macro_f1":f1}

# dataset and balanced sampler
class TextDS(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=512):
        self.texts=texts; self.labels=labels; self.tok=tokenizer; self.max_length=max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self,idx):
        enc=self.tok(self.texts[idx],truncation=True,max_length=self.max_length)
        item={k:torch.tensor(v) for k,v in enc.items()}
        item["labels"]=torch.tensor(self.labels[idx],dtype=torch.long)
        return item

def make_weighted_sampler(labels, num_classes):
    counts=np.bincount(labels,minlength=num_classes)
    weights=1.0/np.maximum(counts,1)
    sample_w=np.array([weights[l] for l in labels])
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)

# weighted Trainer with class-balanced sampling
class BalancedWeightedTrainer(Trainer):
    def __init__(self, class_weights=None, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.num_classes = num_classes
    def get_train_dataloader(self):
        sampler=make_weighted_sampler([int(self.train_dataset[i]["labels"]) for i in range(len(self.train_dataset))], self.num_classes)
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, sampler=sampler, collate_fn=self.data_collator)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels=inputs.get("labels")
        outputs=model(**{k:v for k,v in inputs.items() if k!="labels"})
        logits=outputs.get("logits")
        loss_fct=nn.CrossEntropyLoss(weight=self.class_weights)
        loss=loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# LoRA config for transformer
def lora_wrap(model, model_type):
    if model_type=="bert":
        targets=["query","key","value","dense"]
    else:
        targets=["q_lin","k_lin","v_lin","out_lin"]
    cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.1, target_modules=targets, bias="none", task_type="SEQ_CLS")
    return get_peft_model(model,cfg)

# LegalBERT fine-tuning with LoRA"
legal_model_name="nlpaueb/legal-bert-base-uncased"
legal_tok=AutoTokenizer.from_pretrained(legal_model_name,use_fast=True)
legal_model=AutoModelForSequenceClassification.from_pretrained(legal_model_name,num_labels=len(classes)).to(DEVICE)
legal_model=lora_wrap(legal_model,"bert")

cw_counts=np.bincount(y_train,minlength=len(classes))
cw=torch.tensor((len(y_train)/(len(classes)*np.maximum(cw_counts,1))).astype(np.float32),device=DEVICE)

legal_train=TextDS(texts_train,y_train,legal_tok,512)
legal_test=TextDS(texts_test,y_test,legal_tok,512)

legal_args=TrainingArguments(output_dir="./artifacts/legalbert_lora",num_train_epochs=2,per_device_train_batch_size=8,per_device_eval_batch_size=16,learning_rate=2e-5,weight_decay=0.01,evaluation_strategy="epoch",save_strategy="no",logging_steps=50,report_to=[],seed=SEED)

legal_trainer=BalancedWeightedTrainer(model=legal_model,args=legal_args,train_dataset=legal_train,eval_dataset=legal_test,data_collator=DataCollatorWithPadding(tokenizer=legal_tok),compute_metrics=compute_metrics_eval,class_weights=cw,num_classes=len(classes),tokenizer=legal_tok)

legal_trainer.train()

legal_metrics=legal_trainer.evaluate()

legal_preds=np.argmax(legal_trainer.predict(legal_test).predictions,axis=1)

print("LegalBERT_LoRA TEST report")
print(classification_report(y_test,legal_preds,target_names=[id_to_class[i] for i in range(len(classes))],digits=4))

# DistilBERT fine-tuning with LoRA"
distil_model_name="distilbert-base-uncased"
distil_tok=AutoTokenizer.from_pretrained(distil_model_name,use_fast=True)
distil_model=AutoModelForSequenceClassification.from_pretrained(distil_model_name,num_labels=len(classes)).to(DEVICE)
distil_model=lora_wrap(distil_model,"distil")

cw=torch.tensor((len(y_train)/(len(classes)*np.maximum(cw_counts,1))).astype(np.float32),device=DEVICE)
distil_train=TextDS(texts_train,y_train,distil_tok,512)
distil_test=TextDS(texts_test,y_test,distil_tok,512)

distil_args=TrainingArguments(output_dir="./artifacts/distilbert_lora",num_train_epochs=2,per_device_train_batch_size=8,per_device_eval_batch_size=16,learning_rate=2e-5,weight_decay=0.01,evaluation_strategy="epoch",save_strategy="no",logging_steps=50,report_to=[],seed=SEED)

distil_trainer=BalancedWeightedTrainer(model=distil_model,args=distil_args,train_dataset=distil_train,eval_dataset=distil_test,data_collator=DataCollatorWithPadding(tokenizer=distil_tok),compute_metrics=compute_metrics_eval,class_weights=cw,num_classes=len(classes),tokenizer=distil_tok)

distil_trainer.train()

distil_metrics=distil_trainer.evaluate()

distil_preds=np.argmax(distil_trainer.predict(distil_test).predictions,axis=1)

print("DistilBERT_LoRA TEST report")
print(classification_report(y_test,distil_preds,target_names=[id_to_class[i] for i in range(len(classes))],digits=4))

# SVD features for GCN"
tfidf=TfidfVectorizer(min_df=2,ngram_range=(1,2),max_features=100000)
Xtr_tfidf=tfidf.fit_transform(texts_train)

svd=TruncatedSVD(n_components=300,random_state=SEED)
Xtr_svd=svd.fit_transform(Xtr_tfidf)

scaler=StandardScaler(with_mean=True,with_std=True)
Xtr=scaler.fit_transform(Xtr_svd)
Xte=scaler.transform(svd.transform(tfidf.transform(texts_test)))

# LoRA-wrapped GCN and training with class-weighted loss"
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_features,in_features))
        nn.init.kaiming_uniform_(self.weight,a=np.sqrt(5))
        self.r=r; self.alpha=alpha; self.scaling=alpha/r
        self.A=nn.Parameter(torch.zeros(r,in_features))
        self.B=nn.Parameter(torch.zeros(out_features,r))
        nn.init.kaiming_uniform_(self.A,a=np.sqrt(5))
        nn.init.zeros_(self.B)
        self.dp=nn.Dropout(dropout)
        self.weight.requires_grad=False
    def forward(self, x):
        w_eff=self.weight + self.B @ self.A * self.scaling
        return self.dp(x) @ w_eff.t()

class SimpleGCN(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout=0.2,r=8,alpha=16):
        super().__init__()
        self.lin1=LoRALinear(in_dim,hid_dim,r=r,alpha=alpha,dropout=dropout)
        self.lin2=LoRALinear(hid_dim,out_dim,r=r,alpha=alpha,dropout=dropout)
        self.dp=nn.Dropout(dropout)
    def forward(self,X,Ahat):
        H=torch.relu(self.lin1(Ahat @ X))
        H=self.dp(H)
        Z=self.lin2(Ahat @ H)
        return Z

def normalize_adj(A):
    A=A+np.eye(A.shape[0],dtype=np.float32)
    d=np.array(A.sum(1)).flatten()
    dinv=np.power(d,-0.5,where=d>0)
    D=np.diag(dinv)
    return D @ A @ D

nbrs=NearestNeighbors(n_neighbors=10,metric="cosine").fit(Xtr)
dist, idx=nbrs.kneighbors(Xtr)
A=np.zeros((len(Xtr),len(Xtr)),dtype=np.float32)
for i in range(len(Xtr)):
    for j,d in zip(idx[i],dist[i]):
        if i==j: continue
        A[i,j]=1.0-d
A_tr=normalize_adj(A).astype(np.float32)

dte, idxte=nbrs.kneighbors(Xte)
A_te=np.zeros((len(Xte),len(Xtr)),dtype=np.float32)
for i in range(len(Xte)):
    for j,d in zip(idxte[i],dte[i]):
        A_te[i,j]=1.0-d
Dtr=np.power(A_tr.sum(1),-0.5,where=A_tr.sum(1)>0)
Dte=np.power(A_te.sum(1),-0.5,where=A_te.sum(1)>0)
A_te_norm=(Dte[:,None]*A_te)*Dtr[None,:]

Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=DEVICE)
Xte_t=torch.tensor(Xte,dtype=torch.float32,device=DEVICE)

A_tr_t=torch.tensor(A_tr,dtype=torch.float32,device=DEVICE)
A_te_t=torch.tensor(A_te_norm,dtype=torch.float32,device=DEVICE)

ytr_t=torch.tensor(y_train,dtype=torch.long,device=DEVICE)
yte_t=torch.tensor(y_test,dtype=torch.long,device=DEVICE)

gcn=SimpleGCN(in_dim=Xtr.shape[1],hid_dim=256,out_dim=len(classes),dropout=0.2,r=8,alpha=16).to(DEVICE)
counts=np.bincount(y_train,minlength=len(classes))
w=torch.tensor((len(y_train)/(len(classes)*np.maximum(counts,1))).astype(np.float32),device=DEVICE)
opt=AdamW(gcn.parameters(),lr=1e-3,weight_decay=1e-4)
crit=nn.CrossEntropyLoss(weight=w)

# GCN LoRA training"
gcn.train()
for epoch in range(40):
    opt.zero_grad()
    logits=gcn(Xtr_t,A_tr_t)
    loss=crit(logits,ytr_t)
    loss.backward()
    opt.step()

# GCN inference and metrics
gcn.eval()
with torch.no_grad():
    H=torch.relu(gcn.lin1(A_tr_t @ Xtr_t))
    H=gcn.dp(H)
    Z_te=gcn.lin2(A_te_t @ H)
    gcn_preds=torch.argmax(Z_te,dim=1).cpu().numpy()
gcn_report=classification_report(y_test,gcn_preds,target_names=[id_to_class[i] for i in range(len(classes))],digits=4)

print("GCN_LoRA TEST report")
print(gcn_report)

# metrics

def macro_metrics(y_true,y_pred):
    rep=classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    return {"accuracy":rep["accuracy"],"macro_precision":rep["macro avg"]["precision"],"macro_recall":rep["macro avg"]["recall"],"macro_f1":rep["macro avg"]["f1-score"]}

m_legal=macro_metrics(y_test,legal_preds)
m_distil=macro_metrics(y_test,distil_preds)
m_gcn=macro_metrics(y_test,gcn_preds)

print({"LegalBERT_LoRA":m_legal, # 93%
       "DistilBERT_LoRA":m_distil, # 92%
       "GCN_LoRA":m_gcn}) # 89%