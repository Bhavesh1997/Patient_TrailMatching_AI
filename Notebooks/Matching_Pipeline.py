import os
import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DATA_DIR = "data/processed"

PATIENT_FILE = os.path.join(DATA_DIR, "cleaned_augmented_patients.csv")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "trial_meta.pkl")


# -----------------------------
# LOAD
# -----------------------------
def load_data():
    patients = pd.read_csv(PATIENT_FILE)
    patients.fillna("", inplace=True)
    return patients


def load_index():
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        trials = pickle.load(f)

    return index, pd.DataFrame(trials)


# -----------------------------
# TEXT CREATION
# -----------------------------
def create_patient_text(row):
    return f"Age: {row['age']}, Gender: {row['GENDER']}, Condition: {row['conditions_text']}"


# -----------------------------
# EMBEDDING
# -----------------------------
def get_embedding(text, model):
    emb = model.encode([text]).astype("float32")
    faiss.normalize_L2(emb)
    return emb


# -----------------------------
# RULE SCORING
# -----------------------------
def compute_rule_score(patient, trial):
    score = 0
    reasons = []

    p_cond = str(patient["conditions_text"]).lower()
    t_cond = str(trial["conditions_text"]).lower()
    t_text = str(trial.get("eligibility_clean", "")).lower()

    if p_cond in t_cond:
        score += 2
        reasons.append("Exact condition match")
    elif any(word in t_cond for word in p_cond.split()):
        score += 1
        reasons.append("Partial match")

    try:
        min_age = float(trial.get("min_age", 0))
        max_age = float(trial.get("max_age", 120))
        if min_age <= patient["age"] <= max_age:
            score += 1
            reasons.append("Age eligible")
        else:
            score -= 1
            reasons.append("Age mismatch")
    except:
        pass

    if trial.get("gender", "All") in ["All", patient["GENDER"]]:
        score += 0.5
        reasons.append("Gender match")
    else:
        score -= 0.5
        reasons.append("Gender mismatch")

    if "no prior cancer" in t_text and "cancer" in p_cond:
        score -= 2
        reasons.append("Exclusion conflict")

    return score, reasons


# -----------------------------
# LLM (LOCAL - FLAN T5)
# -----------------------------
print("Loading local LLM...")
llm = pipeline("text2text-generation", model="google/flan-t5-base")


def llm_evaluate(patient, trial):
    prompt = f"""
Patient:
Age: {patient['age']}
Gender: {patient['GENDER']}
Condition: {patient['conditions_text']}

Trial Eligibility:
{trial.get('eligibility_clean', '')}

Question:
Is this patient eligible? Answer yes or no with a short reason.
"""

    try:
        output = llm(prompt, max_length=80)[0]["generated_text"].lower()

        if "yes" in output:
            return 1, output
        else:
            return -1, output

    except:
        return 0, "LLM failed"


# -----------------------------
# MATCHING
# -----------------------------
def match_patient(patient, index, trials, model, top_k=10):

    query = create_patient_text(patient)
    emb = get_embedding(query, model)

    D, I = index.search(emb, top_k)

    results = []

    for rank, idx in enumerate(I[0]):
        trial = trials.iloc[idx]

        sim_score = float(D[0][rank])
        rule_score, reasons = compute_rule_score(patient, trial)

        # LLM only for top 3 (performance)
        if rank < 3:
            llm_score, llm_reason = llm_evaluate(patient, trial)
        else:
            llm_score, llm_reason = 0, "Skipped"

        final_score = (
            0.6 * sim_score +
            0.3 * rule_score +
            0.1 * llm_score
        )

        results.append({
            "patient_id": patient["patient_id"],
            "trial_id": trial["nct_id"],
            "condition": trial["conditions_text"],
            "final_score": final_score,
            "similarity": sim_score,
            "rule_score": rule_score,
            "llm_score": llm_score,
            "reasons": ", ".join(reasons),
            "llm_reason": llm_reason
        })

    return pd.DataFrame(results).sort_values(by="final_score", ascending=False)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading data...")
    patients = load_data()

    print("Loading FAISS index...")
    index, trials = load_index()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nRunning matching...\n")

    for i in range(min(3, len(patients))):

        patient = patients.iloc[i]

        print("\n==============================")
        print("Patient:", patient["patient_id"], "|", patient["conditions_text"])

        result = match_patient(patient, index, trials, model)

        print(result.head(3)[["trial_id", "final_score", "llm_reason"]])