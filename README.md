# Navigating-the-Rulespace
Capstone Project collaborated with Bloomberg Government


## 🔎 Executive Summary
- **Data**: Bloomberg Government provided a curated set of congressional bills and inline citations (≈108 bills, 17k+ citations).  
- **Challenge**: No gorund truth labels; strong class imbalance.  
- **Solution**: Multi-stage ground-truth pipeline, then comparative modeling, and a lightweight chatbot configured in Azure AI Foundry.  
- **Highlights**
  - **Label quality**: Created the ground truth labeling process - manual + LLM +  ML training
  - **Classification**: ~95% accuracy overall; ~0.93 macro-F1 / ~93% domain precision after LoRA fine-tuning.  
  - **Analyst workflow**: Helped legal analyst by reduving manual review time.


## 📦 Data & Label Taxonomy
- **Source**: Congressional bills with inline citations provided by sponsor.  
- **Target Labels (5)**: `Authority`, `Amending`, `Definition`, `Exception`, `Precedent`.  
- **Schema**: `(bill_id, section_id, citation_text, context_pre, context, context_post, jurisdiction, year, metadata…)`  
- **Normalization**: regex cleanup, unicode fixes, citation canonicalization, and 2–3 sentence context windows.

---

## 🏷️ Ground-Truth Generation (multi-stage)
1) **Expert gold**: **~20%** of corpus manually annotated by legal analysts. 
2) **LLM seeding**: chain-of-thought prompt based classification with Meta **LLaMA** propose a label + short rationale.  
   - After generating goes thorugh a review process of expert in the loop.
   - Accepted items become silver labels. Rationales are **not used** at inference time.
3) **LLM + expert verify**: **~35%** labeled using LLM and with **legal expert in the loop** to validate/rectify.  
4) **Model-assisted + expert verify**: Train **Legal RoBERTa** on the first **~55%**; label the remaining **~45%** using model's prediciton which are again gone though expert verification process  
5) **Class balance**: **SMOTE** on sentence embeddings + **frozen adapters** for stable training batches.

---

## 🧪 Models & Training
- **Encoders**: **LegalBERT**, **DistilBERT**.  
- **Graph context**: **GCN** over a citation graph (nodes=citations; edges=co-mentions/cross-refs) using encoder embeddings + simple structural features.  
- **Fine-tuning**: **LoRA** adapters for parameter-efficient updates (fast iteration on modest GPUs).  
- **Splits**: Held-out by bill and year to avoid near-duplicate leakage.  
- **Metrics**: Accuracy, macro-Precision/Recall/F1; per-class confusion; calibration curves.

---
