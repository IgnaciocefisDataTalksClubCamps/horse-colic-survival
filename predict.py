"""Web service that predicts the survival probability of a horse with colic.

Run it with:

    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
"""

import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from preprocessing import clean, to_dicts

MODEL_PATH = "model.bin"


class Horse(BaseModel):
    """Clinical data of a horse on admission.

    Categorical features are optional: a missing value is treated as "unknown",
    as in training. Numerical features are required, with plausible ranges.
    """

    # Reject unknown fields, so a typo in a field name does not go unnoticed
    model_config = ConfigDict(extra="forbid")

    # Categorical features (valid values after cleaning)
    age: Literal["adult", "young"] | None = None
    temp_of_extremities: Literal["cold", "cool", "normal", "warm"] | None = None
    peripheral_pulse: Literal["absent", "reduced", "normal", "increased"] | None = None
    mucous_membrane: Literal[
        "normal_pink", "bright_pink", "pale_pink", "pale_cyanotic", "bright_red", "dark_cyanotic"
    ] | None = None
    capillary_refill_time: Literal["less_3_sec", "more_3_sec"] | None = None
    pain: Literal["alert", "depressed", "mild_pain", "severe_pain", "extreme_pain"] | None = None
    peristalsis: Literal["hypermotile", "normal", "hypomotile", "absent"] | None = None
    abdominal_distention: Literal["none", "slight", "moderate", "severe"] | None = None
    nasogastric_tube: Literal["none", "slight", "significant"] | None = None
    nasogastric_reflux: Literal["none", "less_1_liter", "more_1_liter"] | None = None
    rectal_exam_feces: Literal["normal", "increased", "decreased", "absent"] | None = None
    abdomen: Literal["normal", "other", "firm", "distend_small", "distend_large"] | None = None
    abdomo_appearance: Literal["clear", "cloudy", "serosanguious"] | None = None

    # Numerical features (ranges a bit wider than in the training data)
    rectal_temp: float = Field(..., ge=30, le=45, description="Rectal temperature (°C)")
    pulse: float = Field(..., ge=20, le=250, description="Heart rate (beats per minute)")
    respiratory_rate: float = Field(..., ge=4, le=120, description="Breaths per minute")
    nasogastric_reflux_ph: float = Field(..., ge=0, le=14, description="pH of the nasogastric reflux")
    packed_cell_volume: float = Field(..., ge=10, le=90, description="Packed cell volume (%)")
    total_protein: float = Field(
        ..., ge=1, le=120, description="Total protein, in g/dL or g/L (values above 30 are converted from g/L)"
    )
    abdomo_protein: float = Field(..., ge=0, le=15, description="Abdominocentesis total protein (g/dL)")


class PredictResponse(BaseModel):
    survival_probability: float
    lived: bool


app = FastAPI(title="horse-colic-survival")

# Load the trained pipeline (DictVectorizer + random forest) once, at startup
with open(MODEL_PATH, "rb") as f_in:
    model = pickle.load(f_in)


def predict_single(horse):
    """Return the survival probability of one horse, given as a dictionary."""
    df = clean(pd.DataFrame([horse]))
    return float(model.predict_proba(to_dicts(df))[0, 1])


@app.post("/predict")
def predict(horse: Horse) -> PredictResponse:
    """Predict the survival probability of a horse with colic."""
    probability = predict_single(horse.model_dump())
    return PredictResponse(survival_probability=probability, lived=probability >= 0.5)


@app.get("/health")
def health() -> dict:
    """Check that the service is running."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root URL to the interactive documentation."""
    return RedirectResponse(url="/docs")
