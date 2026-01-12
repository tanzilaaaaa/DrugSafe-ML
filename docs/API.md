# API Documentation

## Drug Interaction Checker API

Base URL: `http://localhost:5001/api`

### Authentication
Currently, no authentication is required. This is an educational project.

---

## Endpoints

### 1. Single Drug Prediction

**Endpoint:** `POST /api/predict`

**Description:** Predict drug interaction between two drugs

**Request:**
```json
{
  "drug1": "Warfarin",
  "drug2": "Aspirin"
}
```

**Response:**
```json
{
  "drug1": "Warfarin",
  "drug2": "Aspirin",
  "interaction": true,
  "interaction_confidence": 0.85,
  "severity": "High",
  "severity_confidence": 0.92,
  "timestamp": "2024-01-06T10:30:00"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (missing parameters)
- `500`: Server error

---

### 2. Batch Prediction

**Endpoint:** `POST /api/batch_predict`

**Description:** Predict interactions for multiple drug pairs

**Request:**
```json
{
  "drug_pairs": [
    {"drug1": "Warfarin", "drug2": "Aspirin"},
    {"drug1": "Metformin", "drug2": "Lisinopril"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "drug1": "Warfarin",
      "drug2": "Aspirin",
      "interaction": true,
      "interaction_confidence": 0.85,
      "severity": "High"
    },
    {
      "drug1": "Metformin",
      "drug2": "Lisinopril",
      "interaction": false,
      "interaction_confidence": 0.78
    }
  ],
  "total_pairs": 2,
  "successful_predictions": 2,
  "timestamp": "2024-01-06T10:30:00"
}
```

---

### 3. Model Information

**Endpoint:** `GET /api/model_info`

**Description:** Get information about loaded models

**Response:**
```json
{
  "models_loaded": true,
  "available_models": {
    "interaction_models": ["LogisticRegression", "DecisionTree", "RandomForest"],
    "severity_models": ["LogisticRegression", "DecisionTree"]
  },
  "dataset_info": {
    "total_records": 1000,
    "interaction_rate": 0.35,
    "unique_drugs": 15
  },
  "timestamp": "2024-01-06T10:30:00"
}
```

---

### 4. Drug List

**Endpoint:** `GET /api/drug_list`

**Description:** Get list of available drugs

**Response:**
```json
{
  "drugs": ["Aspirin", "Warfarin", "Metformin", ...],
  "drug_classes": ["Analgesic", "Anticoagulant", "Antidiabetic", ...],
  "total_drugs": 15,
  "total_classes": 5
}
```

---

### 5. Statistics

**Endpoint:** `GET /api/statistics`

**Description:** Get dataset and model statistics

**Response:**
```json
{
  "dataset": {
    "total_records": 1000,
    "interaction_rate": 0.35,
    "severity_distribution": {
      "Low": 150,
      "Moderate": 200,
      "High": 50
    }
  },
  "model_performance": {
    "interaction": {
      "LogisticRegression": {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85
      }
    }
  }
}
```

---

## Error Handling

All errors return JSON with an error message:

```json
{
  "error": "Both drug names are required"
}
```

Common error codes:
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Endpoint doesn't exist
- `500`: Internal Server Error

---

## Rate Limiting

Currently, no rate limiting is implemented. This is an educational project.

---

## Examples

### Using cURL

```bash
# Single prediction
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"drug1": "Warfarin", "drug2": "Aspirin"}'

# Get drug list
curl http://localhost:5001/api/drug_list

# Get statistics
curl http://localhost:5001/api/statistics
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5001/api/predict',
    json={'drug1': 'Warfarin', 'drug2': 'Aspirin'}
)
print(response.json())

# Batch prediction
response = requests.post(
    'http://localhost:5001/api/batch_predict',
    json={
        'drug_pairs': [
            {'drug1': 'Warfarin', 'drug2': 'Aspirin'},
            {'drug1': 'Metformin', 'drug2': 'Lisinopril'}
        ]
    }
)
print(response.json())
```

---

## Disclaimer

⚠️ **This API is for educational purposes only. Do not use for real medical decisions. Always consult healthcare professionals.**
