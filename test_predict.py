"""Send a sample horse to the prediction service and print the response.

Usage:

    uv run python test_predict.py                       # local service
    uv run python test_predict.py https://<your-app>    # deployed service
"""

import sys

import requests

# Base URL of the service: first command-line argument, or the local service
url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:9696"

# An invented horse with plausible values (not a row of the dataset)
horse = {
    "age": "adult",
    "temp_of_extremities": "normal",
    "peripheral_pulse": "normal",
    "mucous_membrane": "bright_pink",
    "capillary_refill_time": "less_3_sec",
    "pain": "alert",
    "peristalsis": "normal",
    "abdominal_distention": "none",
    "nasogastric_tube": "none",
    "nasogastric_reflux": "none",
    "rectal_exam_feces": "normal",
    "abdomen": "normal",
    "abdomo_appearance": "clear",
    "rectal_temp": 38.5,
    "pulse": 60,
    "respiratory_rate": 20,
    "nasogastric_reflux_ph": 7.0,
    "packed_cell_volume": 35.0,
    "total_protein": 7.0,
    "abdomo_protein": 2.0,
}

# Generous timeout: a free cloud instance can take about a minute to wake up
response = requests.post(f"{url}/predict", json=horse, timeout=120)
response.raise_for_status()
result = response.json()

print(f"Survival probability: {result['survival_probability']:.1%}")
if result["lived"]:
    print("The horse is likely to survive.")
else:
    print("The horse is not likely to survive.")
