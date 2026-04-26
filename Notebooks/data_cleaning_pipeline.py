import ast
import os
import re
from datetime import date, datetime

import pandas as pd


DATA_DIR = "Data"
RAW_PATIENTS_PATH = os.path.join(DATA_DIR, "synthea", "augmented_oncology_patients.csv")
RAW_TRIALS_PATH = os.path.join(DATA_DIR, "trials", "oncology_trials.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
TODAY = date.today()
ONCOLOGY_TERMS = {
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "tumors",
    "tumours",
    "leukemia",
    "lymphoma",
    "sarcoma",
    "melanoma",
    "myeloma",
    "neoplasm",
    "oncology",
    "glioblastoma",
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"[\[\]\(\)\{\}',:/\-]+", " ", text)
    return " ".join(text.split())


def clean_condition_label(value):
    text = " ".join(str(value).strip().split())
    text = re.sub(r"\s+", " ", text)
    return text


def parse_condition_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [clean_condition_label(item) for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [clean_condition_label(item) for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [clean_condition_label(part) for part in text.split(",") if part.strip()]


def is_oncology_condition(condition_name):
    normalized = normalize_text(condition_name)
    return any(term in normalized for term in ONCOLOGY_TERMS)


def parse_age_to_years(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return None

    match = re.match(r"^(\d+(?:\.\d+)?)\s+([A-Za-z]+)", text)
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower()

    if "year" in unit:
        return int(number)
    if "month" in unit:
        return max(0, int(number / 12))
    if "week" in unit or "day" in unit:
        return 0

    return int(number)


def calculate_age(birthdate_value):
    birthdate = datetime.strptime(str(birthdate_value), "%Y-%m-%d").date()
    return TODAY.year - birthdate.year - (
        (TODAY.month, TODAY.day) < (birthdate.month, birthdate.day)
    )


def normalize_gender(value):
    text = str(value).strip().upper()
    if text in {"M", "MALE"}:
        return "M"
    if text in {"F", "FEMALE"}:
        return "F"
    return "UNKNOWN"


def normalize_trial_sex(value):
    text = str(value).strip().upper()
    if text in {"M", "MALE"}:
        return "MALE"
    if text in {"F", "FEMALE"}:
        return "FEMALE"
    return "ALL"


def split_eligibility_text(text):
    if pd.isna(text):
        return "", "", ""

    cleaned = re.sub(r"\r\n?", "\n", str(text))
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned).strip()

    lower = cleaned.lower()
    inclusion = ""
    exclusion = ""

    inclusion_start = lower.find("inclusion criteria")
    exclusion_start = lower.find("exclusion criteria")

    if inclusion_start != -1 and exclusion_start != -1:
        if inclusion_start < exclusion_start:
            inclusion = cleaned[inclusion_start:exclusion_start].strip()
            exclusion = cleaned[exclusion_start:].strip()
        else:
            exclusion = cleaned[exclusion_start:inclusion_start].strip()
            inclusion = cleaned[inclusion_start:].strip()
    elif inclusion_start != -1:
        inclusion = cleaned[inclusion_start:].strip()
    elif exclusion_start != -1:
        exclusion = cleaned[exclusion_start:].strip()

    return cleaned, inclusion, exclusion


def clean_augmented_patients(input_path=RAW_PATIENTS_PATH):
    patients = pd.read_csv(input_path).copy()
    patients["patient_id"] = patients["patient_id"].astype(str).str.strip()
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["GENDER"] = patients["GENDER"].apply(normalize_gender)
    patients["condition"] = patients["condition"].apply(clean_condition_label)
    patients = patients.dropna(subset=["patient_id", "BIRTHDATE", "condition"])
    patients = patients.drop_duplicates(subset=["patient_id", "condition"])
    patients = patients[patients["condition"].map(is_oncology_condition)].copy()

    patient_level = (
        patients.sort_values(["patient_id", "condition"])
        .groupby("patient_id", as_index=False)
        .agg(
            BIRTHDATE=("BIRTHDATE", "first"),
            GENDER=("GENDER", "first"),
            conditions=("condition", list),
        )
    )

    patient_level["age"] = patient_level["BIRTHDATE"].dt.date.apply(calculate_age)
    patient_level["condition_count"] = patient_level["conditions"].apply(len)
    patient_level["conditions_text"] = patient_level["conditions"].apply(" | ".join)
    patient_level["primary_condition"] = patient_level["conditions"].apply(
        lambda values: values[0] if values else ""
    )
    patient_level["BIRTHDATE"] = patient_level["BIRTHDATE"].dt.strftime("%Y-%m-%d")

    return patient_level[
        [
            "patient_id",
            "BIRTHDATE",
            "age",
            "GENDER",
            "primary_condition",
            "condition_count",
            "conditions",
            "conditions_text",
        ]
    ]


def clean_trials(input_path=RAW_TRIALS_PATH):
    trials = pd.read_csv(input_path).copy()
    trials = trials.drop_duplicates(subset=["nct_id"])
    trials["sex"] = trials["sex"].apply(normalize_trial_sex)
    trials["min_age_years"] = trials["min_age"].apply(parse_age_to_years)
    trials["max_age_years"] = trials["max_age"].apply(parse_age_to_years)
    trials["conditions_list"] = trials["conditions"].apply(parse_condition_list)
    trials["conditions_list"] = trials["conditions_list"].apply(
        lambda values: [value for value in values if is_oncology_condition(value)]
    )
    trials = trials[trials["conditions_list"].map(bool)].copy()
    trials["conditions_text"] = trials["conditions_list"].apply(" | ".join)

    eligibility_parts = trials["eligibility"].apply(split_eligibility_text)
    trials["eligibility_clean"] = eligibility_parts.apply(lambda item: item[0])
    trials["inclusion_criteria"] = eligibility_parts.apply(lambda item: item[1])
    trials["exclusion_criteria"] = eligibility_parts.apply(lambda item: item[2])

    trials["status"] = trials["status"].astype(str).str.strip().str.upper()
    trials["phase"] = trials["phase"].fillna("UNKNOWN").astype(str)
    trials["healthy_volunteers"] = (
        trials["healthy_volunteers"].fillna("UNKNOWN").astype(str).str.upper()
    )
    trials["match_text"] = (
        trials["title"].astype(str).str.strip()
        + " | "
        + trials["conditions_text"]
        + " | "
        + trials["eligibility_clean"].astype(str).str.slice(0, 1500)
    )

    return trials[
        [
            "nct_id",
            "title",
            "status",
            "phase",
            "sex",
            "healthy_volunteers",
            "min_age_years",
            "max_age_years",
            "conditions_list",
            "conditions_text",
            "eligibility_clean",
            "inclusion_criteria",
            "exclusion_criteria",
            "match_text",
        ]
    ]


def save_cleaned_data(
    patients_df,
    trials_df,
    output_dir=OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    patients_path = os.path.join(output_dir, "cleaned_augmented_patients.csv")
    trials_path = os.path.join(output_dir, "cleaned_oncology_trials.csv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        patients_df.to_csv(patients_path, index=False)
    except PermissionError:
        patients_path = os.path.join(
            output_dir,
            f"cleaned_augmented_patients_{timestamp}.csv",
        )
        patients_df.to_csv(patients_path, index=False)

    try:
        trials_df.to_csv(trials_path, index=False)
    except PermissionError:
        trials_path = os.path.join(
            output_dir,
            f"cleaned_oncology_trials_{timestamp}.csv",
        )
        trials_df.to_csv(trials_path, index=False)

    return patients_path, trials_path


def run_cleaning_pipeline():
    patients_df = clean_augmented_patients()
    trials_df = clean_trials()
    patients_path, trials_path = save_cleaned_data(patients_df, trials_df)

    summary = {
        "cleaned_patient_rows": len(patients_df),
        "cleaned_trial_rows": len(trials_df),
        "cleaned_patient_path": patients_path,
        "cleaned_trial_path": trials_path,
    }

    return patients_df, trials_df, summary


if __name__ == "__main__":
    patients_df, trials_df, summary = run_cleaning_pipeline()
    print("Cleaning complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
