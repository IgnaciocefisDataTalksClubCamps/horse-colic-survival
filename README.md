# Horse Colic Survival Prediction

A machine learning web service that estimates the probability that a horse with colic will survive, using information available when it is admitted to a veterinary hospital.

> **Project status:** in progress. Midterm project for [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). Sections marked *TBD* will be completed as the work advances.

## Problem

Colic (abdominal pain) is a serious and potentially fatal condition in horses. On admission, veterinarians must quickly assess how severe a case is and discuss the options with the owner.

This project trains a model on clinical data such as vital signs, pain level and examination findings. It exposes the model through an API: a client sends a horse's data and receives a survival probability.

The score is meant to support the veterinarian's judgement, not to replace it.

## Dataset

The project uses the training data from the Kaggle competition [Predict Health Outcomes of Horses](https://www.kaggle.com/competitions/playground-series-s3e22) (Playground Series S3E22). This is a synthetic dataset derived from the real [Horse Colic dataset](https://archive.ics.uci.edu/dataset/47/horse+colic) from UCI, which has 368 cases.

- **Size:** 1,235 labelled rows and 29 columns.
- **Original target:** `outcome`, with the values `lived`, `died` and `euthanized`.
- **Project target:** `lived` versus `not lived` (`died` + `euthanized`).
- **Why binary:** the service estimates survival, and euthanasia is partly a human decision rather than a purely clinical outcome.
- **Missing values:** 464 rows have missing values, all in categorical columns. The CSV stores them as the text `None`, which pandas reads as missing by default. This is different from lowercase `none`, a valid category in columns such as `abdominal_distention`.
- **Repeated cases:** `hospital_number` has 255 unique values across the 1,235 rows, and 198 of them appear more than once. The data is synthetic, so a repeated value does not necessarily mean the same horse: 161 of those 198 values have more than one outcome.

### Excluded columns

| Columns | Reason |
|---|---|
| `id`, `hospital_number` | Identifiers (`hospital_number` is not a model feature, but it is used to group rows when splitting the data) |
| `cp_data` | UCI describes it as having no significance |
| `surgical_lesion`, `lesion_1`, `lesion_2`, `lesion_3` | Only known after surgery or autopsy, so they would leak future information |

Whether to keep `surgery` will be decided by comparing models with and without it, since it is a treatment decision rather than an admission measurement.

## Approach

1. **Data preparation:** clean the data, handle missing values and remove columns that leak information.
2. **Exploratory data analysis:** study value ranges, missing values, the target distribution and feature importance.
3. **Model training:** compare logistic regression, a decision tree, a random forest and XGBoost, tuning each one with stratified 5-fold cross-validation grouped by `hospital_number`, so that the same value never appears in both the training and validation folds.
4. **Model selection:** choose a model using ROC AUC, with accuracy and F1 as supporting metrics, then evaluate it once on a 20% held-out test set that shares no `hospital_number` values with the training data.
5. **Deployment:** serve the final model with FastAPI, package it with Docker and deploy it to the cloud.

## Results

*TBD: model comparison table and final test score.*

## Requirements

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/), which manages Python and the project dependencies
- [Docker](https://www.docker.com/), to run the service in a container
- A [Kaggle](https://www.kaggle.com/) account, to download the data (an API token is only needed for the command-line download)

## How to run

### 1. Install the dependencies

```bash
git clone git@github.com:IgnaciocefisDataTalksClubCamps/horse-colic-survival.git
cd horse-colic-survival
uv sync --locked
```

`uv sync` creates the virtual environment in `.venv`. You can prefix commands with `uv run` as shown below, so you don't need to activate it. To activate it manually:

```bash
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```

### 2. Download the data

The data is not included in this repository because the competition rules (section 7.B) do not allow redistributing it.

Accept the [competition rules](https://www.kaggle.com/competitions/playground-series-s3e22/rules), configure your Kaggle API token, and run:

```bash
uv run kaggle competitions download -c playground-series-s3e22 -p data --unzip
```

Alternatively, download the files from the competition's [Data page](https://www.kaggle.com/competitions/playground-series-s3e22/data) without an API token and extract them into `data/`. The training file must end up at `data/train.csv`, not in a subfolder.

### 3. Train the model

```bash
uv run python train.py
```

This saves the trained model to `model.bin`.

### 4. Run the service locally

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

### 5. Run the service with Docker

```bash
docker build -t horse-colic-survival .
docker run -it --rm -p 9696:9696 horse-colic-survival
```

## Using the API

With the service running, you can use either of these options:

- Open the interactive documentation at <http://localhost:9696/docs>.
- Send a sample request with the test script:

```bash
uv run python test_predict.py
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Takes a horse's clinical data and returns the survival probability |
| `/health` | GET | Checks that the service is running |

## Cloud deployment

*TBD: deployment instructions and a public URL, video or screenshot of the running service.*

## Project structure

```text
├── README.md
├── notebook.ipynb     # Data preparation, EDA and model selection
├── train.py           # Trains and saves the final model
├── predict.py         # FastAPI web service
├── test_predict.py    # Sample request to the service
├── pyproject.toml     # Dependencies
├── uv.lock            # Locked dependency versions
├── .python-version    # Python version
├── Dockerfile         # Container image for the service
├── model.bin          # Trained model
└── data/              # Dataset (downloaded from Kaggle)
```

## Limitations

- The data is synthetic, so the model is not clinically validated and must not be used for real veterinary decisions.
- The dataset is small, so performance estimates may vary between data splits.
- The target groups `died` and `euthanized` together, so it does not separate clinical death from the decision to euthanize.
