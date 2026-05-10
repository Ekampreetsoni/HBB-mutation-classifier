# 🧬 HBB Mutation Classifier

**Sequence-based and annotation-based pathogenicity prediction for beta-globin gene variants**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)](https://streamlit.io)
[![ClinVar](https://img.shields.io/badge/Data-ClinVar-green)](https://www.ncbi.nlm.nih.gov/clinvar/)
[![Ensembl](https://img.shields.io/badge/Sequences-Ensembl_REST-purple)](https://rest.ensembl.org)

---

## Overview

This project classifies **HBB (beta-globin) gene mutations** as **pathogenic** or **benign** using machine learning and bioinformatics. It was built to explore computational approaches for genetic variant interpretation relevant to **beta-thalassemia** and related hemoglobinopathies — conditions with high prevalence in India and globally.

The project implements two complementary classification pipelines:

| Pipeline | Features | Model | AUC |
|---|---|---|---|
| **Annotation-based** | Variant type, molecular consequence (one-hot) | Random Forest | ~0.92 |
| **Sequence-based** | Delta k-mer frequencies (3-mer, ±20 bp window) | Random Forest | dataset-dependent |

---

## Project Structure

```
hbb-mutation-classifier/
│
├── data/
│   └── clinvar_result.txt          # ClinVar HBB export (TSV)
│
├── src/
│   ├── preprocess.py               # Data loading, filtering, label creation, encoding
│   ├── train_annotation_model.py   # Annotation-based RF pipeline
│   └── sequence_pipeline.py        # Ensembl API, mutation application, k-mer encoding
│
├── notebooks/
│   ├── 01_annotation_model.ipynb   # Step-by-step annotation pipeline
│   └── 02_sequence_model.ipynb     # DNA sequence extraction and k-mer modelling
│
├── results/                        # Saved plots, metrics, cached numpy arrays
│
├── app.py                          # Streamlit interactive demo
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/hbb-mutation-classifier.git
cd hbb-mutation-classifier
pip install -r requirements.txt
```

### 2. Run the annotation model

```bash
cd src
python train_annotation_model.py --data ../data/clinvar_result.txt
```

Outputs plots and metrics to `results/`.

### 3. Run the sequence pipeline

> Requires internet access (calls the Ensembl REST API)

```bash
python sequence_pipeline.py --data ../data/clinvar_result.txt --max_variants 200
```

### 4. Launch the interactive demo

```bash
streamlit run app.py
```

### 5. Explore the notebooks

```bash
jupyter notebook notebooks/
```

---

## Biological Background

The **HBB gene** encodes the beta-globin subunit of haemoglobin. Pathogenic variants cause **beta-thalassemia**, one of the most common inherited blood disorders worldwide, particularly prevalent across South Asia, the Mediterranean, and the Middle East.

Key mutation categories:
- **Frameshift / nonsense** → loss of functional protein → severe disease
- **Missense** → altered protein → variable severity
- **Splice site** → aberrant mRNA → partial/complete dysfunction
- **Promoter / intronic** → reduced expression → mild to moderate disease

---

## Pipeline Details

### Annotation-Based Pipeline

1. Load ClinVar TSV export
2. Filter for HBB gene variants
3. Binarise `Germline classification` → Pathogenic (1) / Benign (0)
4. One-hot encode `Variant type` and `Molecular consequence`
5. Random oversampling to balance classes
6. Train Random Forest (n=100, class_weight='balanced')
7. Evaluate: accuracy, F1, ROC-AUC, confusion matrix

**Key finding:** Frameshift and stop-gained variants are the strongest predictors of pathogenicity.

### Sequence-Based Pipeline

```
ClinVar SPDI field
       ↓
Parse: chromosome, position, REF, ALT
       ↓
Ensembl REST API → 41 bp reference sequence
       ↓
Apply ALT allele → mutated sequence
       ↓
3-mer frequency vectors for both sequences
       ↓
Δ k-mer vector (mutated − reference) → features
       ↓
Random Forest classifier
```

**SPDI example:** `NC_000011.10:5226966:A:T`  
→ chromosome 11, position 5226966, A→T substitution

---

## Results

### Annotation Model

![Annotation model plots](results/annotation_model_plots.png)

| Metric | Value |
|---|---|
| Accuracy | ~85% |
| ROC-AUC | ~0.92 |
| Top features | frameshift, stop gained, synonymous |

### Sequence Model

The sequence model is limited by Ensembl API rate limits and dataset size, but demonstrates the feasibility of using raw sequence context for variant classification.

---

## Technologies

- **Python 3.10+**
- **pandas, numpy** — data manipulation
- **scikit-learn** — Random Forest, metrics
- **matplotlib** — visualisation
- **requests** — Ensembl REST API
- **Streamlit** — interactive demo
- **Ensembl REST API** — genomic sequence retrieval
- **ClinVar** — variant annotation database

---

## Future Work

- [ ] **DNABERT** — transformer embeddings for genomic sequences
- [ ] **Larger sequence window** — ±100 bp for regulatory context
- [ ] **CNN/RNN models** — end-to-end sequence learning
- [ ] **SHAP explainability** — which k-mer shifts drive pathogenicity
- [ ] **Multi-gene extension** — HBA1/HBA2 for alpha-thalassemia
- [ ] **Clinical severity scoring** — beyond binary classification

---

## Data Source

Variants downloaded from [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) with search filter: Gene = HBB.  
Genomic sequences fetched from [Ensembl REST API](https://rest.ensembl.org) (GRCh38/hg38).

---

## Author

Built as part of a computational genomics project exploring machine learning approaches to genetic variant interpretation for haemoglobinopathies relevant to South Asian populations.

---

## License

MIT
