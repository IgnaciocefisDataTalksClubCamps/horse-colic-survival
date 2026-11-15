"""Train the final model and save it to model.bin.

Reproduces the final evaluation of notebook.ipynb: the random forest chosen in the
notebook is trained on train + validation (80% of the data) and evaluated once on
the held-out test set (20%), split by hospital_number groups.
"""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from preprocessing import clean, to_dicts

DATA_PATH = "data/train.csv"
MODEL_PATH = "model.bin"
SEED = 42

# Hyperparameters chosen in the notebook
PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "random_state": SEED,
    "n_jobs": -1,
}


def load_data():
    """Load and clean the data. Return the features, the target and the groups."""
    df = pd.read_csv(DATA_PATH)

    # Binary target: 1 = lived, 0 = died or euthanized
    y = (df["outcome"] == "lived").astype(int)

    # hospital_number is not a feature, but it defines the groups for the split
    groups = df["hospital_number"]

    df = clean(df.drop(columns="outcome"))
    return df, y, groups


def split(df, y, groups):
    """Hold out 20% of the rows as the test set, keeping each group in a single set.

    Same split as step 1 in the notebook. The validation set is not needed here
    because the hyperparameters are already chosen.
    """
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    full_train_idx, test_idx = next(splitter.split(df, y, groups))

    df_full_train = df.iloc[full_train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    y_full_train = y.iloc[full_train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return df_full_train, df_test, y_full_train, y_test


def train_model(df_train, y_train):
    """Train the one-hot encoding + random forest pipeline."""
    model = make_pipeline(
        DictVectorizer(sparse=False),
        RandomForestClassifier(**PARAMS),
    )
    model.fit(to_dicts(df_train), y_train)
    return model


def evaluate_model(model, df_test, y_test):
    """Print ROC AUC, accuracy and F1 on the test set (0.5 threshold)."""
    y_pred = model.predict_proba(to_dicts(df_test))[:, 1]

    print(f"Test ROC AUC:  {roc_auc_score(y_test, y_pred):.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred >= 0.5):.3f}")
    print(f"Test F1:       {f1_score(y_test, y_pred >= 0.5):.3f}")


def save_model(model, path):
    """Save the trained pipeline with pickle."""
    with open(path, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    df, y, groups = load_data()
    df_full_train, df_test, y_full_train, y_test = split(df, y, groups)

    print(f"Training on {len(df_full_train)} rows, testing on {len(df_test)} rows")
    model = train_model(df_full_train, y_full_train)
    evaluate_model(model, df_test, y_test)
    save_model(model, MODEL_PATH)
