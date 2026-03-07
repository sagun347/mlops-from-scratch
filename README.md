# MLOps From Scratch

This project demonstrates a **basic Machine Learning Operations (MLOps) pipeline**.

It trains a machine learning model, versions it, and serves it through an API using **FastAPI** and **Docker**.

The purpose of this project is to understand how machine learning models move from experimentation to **production systems**.

---

# What This Project Does

1. Trains a machine learning model
2. Saves the model with automatic versioning
3. Stores training metrics
4. Serves the model through an API
5. Allows switching between model versions
6. Runs inside a Docker container

---

# Tech Stack

- Python
- Scikit-Learn
- FastAPI
- Docker
- GitHub

---

# Project Structure


mlops-from-scratch
│
├── app
│ └── main.py # FastAPI API service
│
├── scripts
│ └── train.py # model training script
│
├── models
│ ├── v1
│ │ ├── model.pkl
│ │ └── metrics.json
│ └── v2
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md


---

# API Endpoints

## Health Check

GET /health

Checks if the API is running.

---

## Model Info

GET /model-info

Returns the currently loaded model and version.

---

## Model Metrics

GET /metrics

Returns the metrics of the current model.

---

## Available Model Versions

GET /versions

Lists all available model versions.

Example response:


{
"available_models": ["v1", "v2"],
"latest": "v2"
}


---

## Load Specific Model

POST /load-model/{version}

Example:

POST /load-model/v1

---

## Prediction

POST /predict

Example request:


{
"features": [13.2, 20.1, 89.5]
}


Example response:


{
"prediction": 1,
"label": "benign",
"model_version": "v2"
}


---

# Running the Project

Build the Docker image:


docker build -t mlops-from-scratch .


Run the container:


docker run -p 8000:8000 mlops-from-scratch


Open API documentation:


http://localhost:8000/docs


---

# Train a New Model

Run the training script:


python scripts/train.py


This automatically creates a new model version inside the `models` directory.

Example:


models/v1
models/v2
models/v3


Each version contains:


model.pkl
metrics.json


---

# Purpose of This Project

This project demonstrates the **basic infrastructure needed to deploy machine learning models** in production environments.

It covers:

- model training
- model versioning
- API serving
- containerization
- reproducible environments

---

# Author

Siddhant Yadav
