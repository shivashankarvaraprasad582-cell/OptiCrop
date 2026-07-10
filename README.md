# 🌾 OptiCrop

**OptiCrop** is an intelligent, ML-powered crop recommendation system built with Python and Flask. It analyzes key soil nutrient levels and environmental parameters to recommend the most suitable crop for your farming conditions — complete with detailed growing insights for every prediction.

---

## ✨ Features

- 🤖 **ML Crop Recommendation** — Random Forest classifier trained on soil and climate data
- 🌍 **23+ Supported Crops** — Covers cereals, pulses, fruits, vegetables, and cash crops
- 📊 **Detailed Crop Insights** — Per-crop summaries, ideal pH, temperature, rainfall, and growing tips
- 🖼️ **Dynamic Hero Images** — Crop-specific visuals update on every recommendation
- 🔄 **Auto Model Training** — Model trains automatically on first run if no saved model is found
- 🎨 **Responsive Web UI** — Clean Flask-based interface for easy input and results display
- 💾 **Model Persistence** — Trained model and scaler saved as `.pkl` files for fast reuse

---

## 🌱 Supported Crops

| Category | Crops |
|---|---|
| **Cereals** | Rice, Wheat, Maize, Barley, Millet, Paddy |
| **Pulses** | Chickpea, Soybean, Pea |
| **Cash Crops** | Sugarcane, Cotton, Coffee, Tea |
| **Fruits** | Banana, Mango, Apple, Orange, Grapes |
| **Vegetables** | Potato, Tomato, Cabbage, Lettuce, Spinach |

---

## 🧠 How It Works

OptiCrop accepts **7 input parameters** and predicts the optimal crop using a trained Random Forest model:

| Parameter | Description |
|---|---|
| **N** | Nitrogen content in soil (mg/kg) |
| **P** | Phosphorous content in soil (mg/kg) |
| **K** | Potassium content in soil (mg/kg) |
| **Temperature** | Ambient temperature (°C) |
| **Humidity** | Relative humidity (%) |
| **pH** | Soil pH level |
| **Rainfall** | Average rainfall (mm) |

The inputs are scaled with `StandardScaler` before inference to ensure consistent model performance.

---

## 🗂️ Project Structure

```
OptiCrop/
├── app.py                  # Flask application — routes, prediction logic, crop guide
├── data_pipeline.py        # Full pipeline — load data, train, evaluate, and save model
├── train_model.py          # Lightweight training script (quick model save)
├── generate_dataset.py     # Synthetic dataset generator (1200 samples, 25 crops)
├── dataset.csv             # Agricultural training dataset
├── requirements.txt        # Python dependencies
├── models/                 # Auto-created after training
│   ├── crop_recommendation_model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html          # Main web interface
└── static/
    ├── css/                # Custom styles
    ├── js/                 # Frontend scripts
    └── img/                # Crop images (apple, banana, maize, etc.)
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- `pip` or a virtual environment manager

### 1. Clone or Navigate to the Project

```powershell
cd e:\OptiCrop\OptiCrop
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A — Automatic (Recommended)

Simply run the Flask app. If no trained model is found, it will **automatically train and save** the model before serving:

```powershell
python app.py
```

### Option B — Manual Model Training First

For evaluation metrics (accuracy, classification report), train explicitly using the full pipeline:

```powershell
python data_pipeline.py
```

This will:
- Load and clean `dataset.csv`
- Scale features with `StandardScaler`
- Train a `RandomForestClassifier` (200 estimators, 80/20 train-test split)
- Print accuracy and full classification report
- Save `models/crop_recommendation_model.pkl` and `models/scaler.pkl`

Then start the app:

```powershell
python app.py
```

### Access the App

Open your browser at:

```
http://127.0.0.1:5000
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `Flask` | ≥ 2.3 | Web framework |
| `scikit-learn` | ≥ 1.4 | ML model and preprocessing |
| `pandas` | ≥ 2.2 | Data loading and manipulation |
| `numpy` | ≥ 2.0 | Numerical computations |
| `matplotlib` | ≥ 3.8 | Data visualization (for analysis) |
| `seaborn` | ≥ 0.13 | Statistical plots (for analysis) |
| `jupyter` | — | Interactive notebooks |
| `pickle5` | — | Model serialization |

---

## 🧪 Generating a Synthetic Dataset

If you need to regenerate a training dataset (1,200 samples across 25 crop types):

```powershell
python generate_dataset.py
```

> **Note:** The generated dataset uses randomized values for demonstration. For production accuracy, replace `dataset.csv` with real agricultural survey data.

---

## 📁 Model Artifacts

After training, the `models/` directory will contain:

| File | Description |
|---|---|
| `crop_recommendation_model.pkl` | Trained Random Forest classifier |
| `scaler.pkl` | Fitted `StandardScaler` for input normalization |

These files are auto-loaded on each prediction request.

---

## 🔧 Extending the Project

- **Add more crops** — Extend `CROP_GUIDE` in `app.py` and add images to `static/img/`
- **Try other algorithms** — Modify `data_pipeline.py` to compare KNN, Logistic Regression, Decision Tree, or SVM
- **Use real data** — Replace `dataset.csv` with a real-world agricultural dataset (e.g., from [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset))
- **Improve the UI** — Edit `templates/index.html` and `static/css/` for a custom design

---

## 📄 License

This project is provided as an open-source sample implementation and may be freely extended for agricultural research, farm decision support, or educational purposes.

---

*Built with 🐍 Python · 🌿 scikit-learn · 🌐 Flask*
