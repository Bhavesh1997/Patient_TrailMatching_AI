import ast
import os
import random
import uuid
from datetime import date, datetime

import pandas as pd
import requests


# -----------------------------
# CONFIG
# -----------------------------
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
ONCOLOGY_QUERY = "cancer OR oncology OR tumor OR carcinoma OR leukemia"
DATA_DIR = "Data"
TODAY = date.today()
GENERIC_ONCOLOGY_TOKENS = {
    "cancer",
    "carcinoma",
    "tumor",
    "tumours",
    "tumors",
    "disorder",
    "disease",
    "situation",
    "suspected",
    "advanced",
    "solid",
    "cell",
    "cells",
    "non",
    "small",
    "stage",
    "tnm",
}
ONCOLOGY_TERMS = {
    "cancer",
    "carcinoma",
    "tumor",
    "tumours",
    "tumors",
    "leukemia",
    "lymphoma",
    "sarcoma",
    "melanoma",
    "oncology",
    "myeloma",
    "neoplasm",
}


# -----------------------------
# 1. FETCH ONCOLOGY TRIALS
# -----------------------------
def fetch_oncology_trials(max_studies=500):
    """
    Fetch oncology-related trials from the ClinicalTrials.gov API.
    """
    params = {
        "query.term": ONCOLOGY_QUERY,
        "pageSize": max_studies,
        "format": "json",
    }

    response = requests.get(CLINICAL_TRIALS_API, params=params, timeout=60)
    response.raise_for_status()

    studies = response.json().get("studies", [])
    parsed_trials = []

    for study in studies:
        protocol = study.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        conditions = protocol.get("conditionsModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        design = protocol.get("designModule", {})
        status = protocol.get("statusModule", {})

        parsed_trials.append(
            {
                "nct_id": identification.get("nctId"),
                "title": identification.get("briefTitle"),
                "conditions": conditions.get("conditions"),
                "eligibility": eligibility.get("eligibilityCriteria"),
                "min_age": eligibility.get("minimumAge"),
                "max_age": eligibility.get("maximumAge"),
                "sex": eligibility.get("sex"),
                "healthy_volunteers": eligibility.get("healthyVolunteers"),
                "status": status.get("overallStatus"),
                "phase": design.get("phases"),
                "design_aspects": design.get("designInfo"),
            }
        )

    return pd.DataFrame(parsed_trials)


def load_trials_data():
    """
    Load cached trial data if it already exists locally.
    """
    path = os.path.join(DATA_DIR, "trials", "oncology_trials.csv")
    return pd.read_csv(path)


# -----------------------------
# 2. SAVE DATA
# -----------------------------
def save_trials(df):
    path = os.path.join(DATA_DIR, "trials", "oncology_trials.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved trials to {path}")


def save_patient_data(df, filename):
    path = os.path.join(DATA_DIR, "synthea", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved patient data to {path}")


# -----------------------------
# 3. LOAD SYNTHEA DATA
# -----------------------------
def load_synthea_data():
    """
    Load Synthea patient and condition data.
    """
    patients = pd.read_csv(os.path.join(DATA_DIR, "synthea", "patients.csv"))
    conditions = pd.read_csv(os.path.join(DATA_DIR, "synthea", "conditions.csv"))

    df = patients.merge(
        conditions,
        left_on="Id",
        right_on="PATIENT",
        how="left",
    )

    df = df[["Id", "BIRTHDATE", "GENDER", "DESCRIPTION"]].copy()
    df.rename(columns={"Id": "patient_id", "DESCRIPTION": "condition"}, inplace=True)

    return df


# -----------------------------
# 4. FILTER ONCOLOGY PATIENTS
# -----------------------------
def filter_oncology_patients(df):
    """
    Keep only cancer-related patients.
    """
    oncology_terms = [
        "cancer",
        "tumor",
        "carcinoma",
        "leukemia",
        "lymphoma",
        "sarcoma",
        "melanoma",
    ]

    mask = df["condition"].fillna("").str.lower().apply(
        lambda value: any(term in value for term in oncology_terms)
    )

    return df[mask].copy()


# -----------------------------
# 5. ELIGIBILITY-ALIGNED SYNTHETIC PATIENTS
# -----------------------------
def normalize_text(value):
    """
    Normalize text for simple keyword matching.
    """
    if pd.isna(value):
        return ""

    text = str(value).lower()
    for char in "[](){}',-/:":
        text = text.replace(char, " ")
    return " ".join(text.split())


def parse_condition_list(value):
    """
    Convert the stored trial conditions into a list.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [item for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [part.strip() for part in text.split(",") if part.strip()]


def meaningful_tokens(text):
    """
    Extract non-generic tokens for better cancer-site matching.
    """
    return {
        token
        for token in normalize_text(text).split()
        if token and token not in GENERIC_ONCOLOGY_TOKENS
    }


def is_oncology_condition(condition_name):
    """
    Decide whether a trial condition is cancer-related enough to synthesize.
    """
    normalized = normalize_text(condition_name)
    tokens = set(normalized.split())
    return bool(tokens & ONCOLOGY_TERMS)


def parse_age_to_years(value):
    """
    Convert trial age text like '18 Years' into an integer number of years.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.lower() == "n/a":
        return None

    parts = text.split()
    if not parts:
        return None

    try:
        number = float(parts[0])
    except ValueError:
        return None

    unit = parts[1].lower() if len(parts) > 1 else "years"

    if "year" in unit:
        return int(number)
    if "month" in unit:
        return max(0, int(number / 12))
    if "week" in unit:
        return 0
    if "day" in unit:
        return 0

    return int(number)


def infer_trial_gender(trial_row, condition_text):
    """
    Infer patient gender from structured trial sex data and oncology keywords.
    """
    sex_value = str(trial_row.get("sex", "")).strip().lower()
    if sex_value == "female":
        return "F"
    if sex_value == "male":
        return "M"

    female_terms = [
        "ovarian",
        "breast",
        "cervical",
        "endometrial",
        "uterine",
        "fallopian",
        "polycystic ovary",
    ]
    male_terms = ["prostate", "testicular", "penile"]

    normalized_condition = normalize_text(condition_text)
    if any(term in normalized_condition for term in female_terms):
        return "F"
    if any(term in normalized_condition for term in male_terms):
        return "M"

    text = " ".join(
        [
            normalize_text(trial_row.get("title", "")),
            normalize_text(trial_row.get("eligibility", "")),
            normalized_condition,
        ]
    )

    if any(term in text for term in female_terms):
        return "F"
    if any(term in text for term in male_terms):
        return "M"

    return None


def format_synthea_condition(condition_name):
    """
    Convert a trial condition name into a Synthea-like condition string.
    """
    text = " ".join(str(condition_name).strip().split())
    if not text:
        return "Oncology disorder (disorder)"

    if "(" in text and ")" in text:
        return text

    if "suspected" in text.lower() or "screening" in text.lower():
        return f"{text} (situation)"

    return f"{text} (disorder)"


def calculate_age_from_birthdate(birthdate_text):
    """
    Calculate age in whole years from a YYYY-MM-DD birthdate.
    """
    birthdate = datetime.strptime(str(birthdate_text), "%Y-%m-%d").date()
    return TODAY.year - birthdate.year - (
        (TODAY.month, TODAY.day) < (birthdate.month, birthdate.day)
    )


def build_seed_patient_library(seed_df):
    """
    Create a patient-level library so we can copy realistic condition bundles.
    """
    library = []
    grouped = seed_df.groupby("patient_id", dropna=False)

    for patient_id, patient_rows in grouped:
        first_row = patient_rows.iloc[0]
        conditions = [
            value for value in patient_rows["condition"].dropna().tolist() if str(value).strip()
        ]

        library.append(
            {
                "patient_id": patient_id,
                "birthdate": first_row["BIRTHDATE"],
                "age": calculate_age_from_birthdate(first_row["BIRTHDATE"]),
                "gender": first_row["GENDER"],
                "conditions": conditions,
                "normalized_conditions": [normalize_text(value) for value in conditions],
            }
        )

    return library


def score_seed_patient(seed_patient, target_conditions, target_gender=None):
    """
    Score how well an existing oncology patient matches a trial condition profile.
    """
    score = 0

    if target_gender and seed_patient["gender"] == target_gender:
        score += 3

    for target in target_conditions:
        normalized_target = normalize_text(target)
        target_tokens = meaningful_tokens(target)

        for condition in seed_patient["normalized_conditions"]:
            condition_tokens = meaningful_tokens(condition)

            if normalized_target and normalized_target in condition:
                score += 6
            if condition and condition in normalized_target:
                score += 4

            overlap = len(target_tokens & condition_tokens)
            score += overlap * 3

    return score


def choose_seed_patient(seed_library, target_conditions, target_gender, rng):
    """
    Pick the best-matching oncology seed patient, with a random tie break.
    """
    scored = []
    for patient in seed_library:
        scored.append(
            (
                score_seed_patient(patient, target_conditions, target_gender),
                rng.random(),
                patient,
            )
        )

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    if not scored:
        return None

    best_score = scored[0][0]
    if best_score <= 0:
        return rng.choice(seed_library)

    best_patients = [patient for score, _, patient in scored if score == best_score]
    return rng.choice(best_patients)


def generate_birthdate(min_age=None, max_age=None, rng=None):
    """
    Generate a birthdate that fits the trial's age range.
    """
    rng = rng or random.Random()

    lower = min_age if min_age is not None else 18
    upper = max_age if max_age is not None else 90

    if upper < lower:
        lower, upper = upper, lower

    sampled_age = rng.randint(lower, upper)
    birth_year = TODAY.year - sampled_age
    birth_month = rng.randint(1, 12)
    birth_day = rng.randint(1, 28)

    return date(birth_year, birth_month, birth_day).isoformat()


def select_relevant_seed_conditions(seed_patient, target_condition):
    """
    Keep only the seed conditions that are relevant to the selected trial condition.
    """
    target_normalized = normalize_text(target_condition)
    target_tokens = meaningful_tokens(target_condition)
    relevant_conditions = []

    for condition in seed_patient["conditions"]:
        normalized_condition = normalize_text(condition)
        condition_tokens = meaningful_tokens(condition)
        overlap = len(target_tokens & condition_tokens)

        if target_normalized and (
            target_normalized in normalized_condition
            or normalized_condition in target_normalized
            or overlap > 0
        ):
            relevant_conditions.append(condition)

    if relevant_conditions:
        return relevant_conditions

    return []


def generate_synthetic_trial_matching_patients(
    trials_df,
    oncology_patients_df,
    patients_per_trial=2,
    random_state=42,
):
    """
    Generate more Synthea-shaped oncology patients that better match trial criteria.

    Output columns intentionally match the existing oncology patient export:
    patient_id, BIRTHDATE, GENDER, condition
    """
    if oncology_patients_df.empty:
        raise ValueError("No oncology patients available to use as seed data.")

    rng = random.Random(random_state)
    seed_library = build_seed_patient_library(oncology_patients_df)
    synthetic_rows = []

    for _, trial in trials_df.iterrows():
        trial_conditions = [
            condition
            for condition in parse_condition_list(trial.get("conditions"))
            if is_oncology_condition(condition)
        ]
        if not trial_conditions:
            continue

        min_age = parse_age_to_years(trial.get("min_age"))
        max_age = parse_age_to_years(trial.get("max_age"))

        for _ in range(patients_per_trial):
            target_condition = rng.choice(trial_conditions)
            target_gender = infer_trial_gender(trial, target_condition)
            seed_patient = choose_seed_patient(
                seed_library,
                [target_condition],
                target_gender,
                rng,
            )

            synthetic_id = str(uuid.uuid4())
            birthdate = generate_birthdate(min_age=min_age, max_age=max_age, rng=rng)
            gender = target_gender or seed_patient["gender"]
            seed_conditions = select_relevant_seed_conditions(seed_patient, target_condition)

            bundle_rows = []
            for condition in seed_conditions:
                bundle_rows.append(
                    {
                        "patient_id": synthetic_id,
                        "BIRTHDATE": birthdate,
                        "GENDER": gender,
                        "condition": condition,
                    }
                )

            existing_normalized = {
                normalize_text(row["condition"]) for row in bundle_rows if row["condition"]
            }

            formatted_condition = format_synthea_condition(target_condition)
            normalized_formatted = normalize_text(formatted_condition)

            if normalized_formatted not in existing_normalized:
                bundle_rows.append(
                    {
                        "patient_id": synthetic_id,
                        "BIRTHDATE": birthdate,
                        "GENDER": gender,
                        "condition": formatted_condition,
                    }
                )

            synthetic_rows.extend(bundle_rows)

    return pd.DataFrame(
        synthetic_rows,
        columns=["patient_id", "BIRTHDATE", "GENDER", "condition"],
    )


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        print("Fetching oncology trials...")
        trials_df = fetch_oncology_trials(500)
        save_trials(trials_df)
    except requests.RequestException as exc:
        print(f"Trial fetch failed, using cached trial file instead: {exc}")
        trials_df = load_trials_data()

    print("Loading Synthea patients...")
    patients_df = load_synthea_data()
    oncology_patients = filter_oncology_patients(patients_df)
    save_patient_data(oncology_patients, "oncology_patients.csv")

    print("Generating eligibility-aligned synthetic patients...")
    synthetic_patients = generate_synthetic_trial_matching_patients(
        trials_df=trials_df,
        oncology_patients_df=oncology_patients,
        patients_per_trial=2,
        random_state=42,
    )

    augmented_patients = pd.concat(
        [oncology_patients, synthetic_patients],
        ignore_index=True,
    )

    save_patient_data(synthetic_patients, "synthetic_trial_matching_patients.csv")
    save_patient_data(augmented_patients, "augmented_oncology_patients.csv")

    print("\nSample Trials:")
    print(trials_df.head())

    print("\nSample Original Oncology Patients:")
    print(oncology_patients.head())

    print("\nSample Synthetic Trial-Matching Patients:")
    print(synthetic_patients.head())
