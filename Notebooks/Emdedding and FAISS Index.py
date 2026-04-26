import os
import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_DIR = "Data/processed"

PATIENT_FILE = os.path.join(DATA_DIR, "cleaned_augmented_patients.csv")
TRIAL_FILE = os.path.join(DATA_DIR, "cleaned_oncology_trials.csv")

INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "trial_meta.pkl")


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data():
    patients = pd.read_csv(PATIENT_FILE)
    trials = pd.read_csv(TRIAL_FILE)

    patients.fillna("", inplace=True)
    trials.fillna("", inplace=True)

    return patients, trials


# -----------------------------
# 2. CREATE TEXT
# -----------------------------
def create_patient_text(row):
    return f"Age: {row['age']} | Gender: {row['GENDER']} | Condition: {row['conditions_text']}"


def create_trial_text(row):
    return f"Title: {row['title']} | Condition: {row['conditions_text']} | Eligibility: {row['eligibility_clean']}"


# -----------------------------
# 3. EMBEDDINGS
# -----------------------------
def generate_embeddings(texts, model):
    return model.encode(texts, show_progress_bar=True)


# -----------------------------
# 4. BUILD FAISS INDEX (COSINE)
# -----------------------------
def build_faiss_index(trial_embeddings):
    trial_embeddings = np.array(trial_embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(trial_embeddings)

    index = faiss.IndexFlatIP(trial_embeddings.shape[1])
    index.add(trial_embeddings)

    return index, trial_embeddings


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    print("Loading data...")
    patients, trials = load_data()

    print("Preparing text...")
    patients["text"] = patients.apply(create_patient_text, axis=1)
    trials["text"] = trials.apply(create_trial_text, axis=1)

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    patient_embeddings = generate_embeddings(patients["text"].tolist(), model)
    trial_embeddings = generate_embeddings(trials["text"].tolist(), model)

    print("Building FAISS index...")
    index, trial_embeddings = build_faiss_index(trial_embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_PATH)

    print("Saving metadata...")
    with open(META_PATH, "wb") as f:
        pickle.dump(trials.to_dict(orient="records"), f)

    print("\nDone.")
    print("Total trials indexed:", index.ntotal)