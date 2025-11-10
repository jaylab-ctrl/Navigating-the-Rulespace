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
