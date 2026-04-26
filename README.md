# 🧬 AI-Powered Clinical Trial Matching System

## 📌 Overview

This project builds an end-to-end AI system to match oncology patients with relevant clinical trials. It combines structured and unstructured healthcare data with modern NLP and vector search techniques to simulate real-world pharma analytics workflows.

---

## 🎯 Problem Statement

Identifying eligible patients for clinical trials is a complex and time-consuming process due to:

* Unstructured eligibility criteria
* Heterogeneous patient data
* Lack of scalable matching systems

This project addresses the problem by building an automated patient–trial matching pipeline.

---

## 🏗️ Project Architecture

Patient Data + Clinical Trials Data
→ Data Cleaning & Standardization
→ Text Embedding (Sentence Transformers)
→ Vector Search (FAISS)
→ Rule-Based Filtering (Age, Gender, Condition)
→ LLM Reasoning Layer (FLAN-T5)
→ Final Ranked Trial Recommendations

---

## 📊 Data Sources

* **ClinicalTrials.gov** – Oncology clinical trial data
* **Synthea** – Synthetic patient dataset
* **PMC (PubMed Central)** – Unstructured clinical text
* **LLM Generated Data** – Synthetic patients based on trial criteria

---

## ⚙️ Features

* Semantic patient–trial matching using embeddings
* Fast similarity search using FAISS
* Hybrid scoring (similarity + rule-based logic)
* Local LLM-based eligibility reasoning (no API required)
* Explainable recommendations with reasoning

---

## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Sentence Transformers
* FAISS (Vector Search)
* Hugging Face Transformers (FLAN-T5)
* NLP & Text Processing

---

## 🚀 How to Run

### 1. Install dependencies

```bash
python -m pip install sentence-transformers faiss-cpu transformers torch pandas
```

### 2. Run embedding pipeline

```bash
python embedding_pipeline.py
```

### 3. Run matching pipeline

```bash
python matching_pipeline.py
```

---

## 📈 Sample Output

For each patient, the system returns:

* Top matching clinical trials
* Matching score
* Eligibility reasoning

Example:

Patient: Lung Cancer, Age 65

Top Matches:

* NCT123 → Score: 0.89 → Eligible
* NCT456 → Score: 0.82 → Partially Eligible

---

## 🧠 Key Learnings

* Applied vector embeddings for healthcare NLP
* Built scalable similarity search using FAISS
* Combined rule-based and AI-based decision systems
* Worked with real-world clinical datasets

---

## 🔮 Future Improvements

* Add hybrid search (BM25 + FAISS)
* Improve eligibility parsing using advanced NLP
* Deploy as a web app (Streamlit)
* Integrate real EHR datasets

---

## 👨‍💻 Author

Bhavesh Nehete
Aspiring Data Scientist | AI/ML Enthusiast

---
## 🧩 Architecture Diagram

```mermaid
flowchart TD

    A[Patient Data<br/>(Synthea + Synthetic + PMC)] --> B[Data Cleaning & Standardization]
    C[Clinical Trials<br/>(ClinicalTrials.gov)] --> B

    B --> D[Feature Engineering<br/>(Structured Patient Profiles)]
    
    D --> E[Text Embedding<br/>(Sentence Transformers)]
    
    E --> F[Vector Database<br/>(FAISS Index)]

    F --> G[Top-K Retrieval<br/>(Cosine Similarity)]

    G --> H[Rule-Based Filtering<br/>(Age, Gender, Condition)]

    H --> I[Scoring & Ranking Layer]

    I --> J[LLM Reasoning<br/>(FLAN-T5 Local Model)]

    J --> K[Final Output<br/>Ranked Clinical Trials + Explanation]
```
