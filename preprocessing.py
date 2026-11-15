"""Data cleaning shared by train.py and predict.py.

It applies the same decisions as the "Data cleaning" section of notebook.ipynb.
"""

# Identifiers, columns without meaning, columns only known after surgery or
# autopsy (data leakage) and `surgery` (a treatment decision, see the notebook)
EXCLUDED_COLUMNS = [
    "id",
    "hospital_number",
    "cp_data",
    "surgical_lesion",
    "lesion_1",
    "lesion_2",
    "lesion_3",
    "surgery",
]

# Values that appear in only one or two rows, artefacts of the synthetic data
RARE_CATEGORIES = {
    "capillary_refill_time": {"3": "more_3_sec"},
    "pain": {"slight": "mild_pain"},
    "nasogastric_reflux": {"slight": "less_1_liter"},
    "peristalsis": {"distend_small": "unknown"},
    "rectal_exam_feces": {"serosanguious": "unknown"},
}

# total_protein mixes g/dL and g/L: values above this threshold are in g/L
PROTEIN_THRESHOLD = 30

CATEGORICAL_COLUMNS = [
    "age",
    "temp_of_extremities",
    "peripheral_pulse",
    "mucous_membrane",
    "capillary_refill_time",
    "pain",
    "peristalsis",
    "abdominal_distention",
    "nasogastric_tube",
    "nasogastric_reflux",
    "rectal_exam_feces",
    "abdomen",
    "abdomo_appearance",
]

NUMERICAL_COLUMNS = [
    "rectal_temp",
    "pulse",
    "respiratory_rate",
    "nasogastric_reflux_ph",
    "packed_cell_volume",
    "total_protein",
    "abdomo_protein",
]

# Features used by the model, in the same order as in the notebook
FEATURES = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


def clean(df):
    """Apply the cleaning decisions of the notebook to a DataFrame.

    Works with any number of rows, including a single horse sent to the API.
    """
    df = df.copy().drop(columns=EXCLUDED_COLUMNS, errors="ignore")

    df = df.replace(RARE_CATEGORIES)

    # Missing values are kept as a category of their own ("unknown"),
    # different from the valid category "none"
    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna("unknown")

    # Convert g/L to g/dL
    df.loc[df["total_protein"] > PROTEIN_THRESHOLD, "total_protein"] /= 10

    return df


def to_dicts(df):
    """Convert the features to the list of dictionaries expected by DictVectorizer."""
    return df[FEATURES].to_dict(orient="records")
