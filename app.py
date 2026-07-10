import os
import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

CROP_GUIDE = {
    "paddy": {
        "summary": "Paddy thrives in warm, water-rich soils and is a staple rice crop for high-yield irrigation systems.",
        "ph": "5.5 - 6.5",
        "temperature": "20 - 30°C",
        "rainfall": "150 - 200 mm",
        "touchpoints": ["Use well-distributed irrigation", "Maintain balanced NPK levels", "Monitor water depth closely"],
        "image": "img/paddy.jpg"
    },
    "maize": {
        "summary": "Maize prefers fertile, well-drained soil and is ideal for warm weather with moderate rainfall.",
        "ph": "5.8 - 7.0",
        "temperature": "18 - 27°C",
        "rainfall": "90 - 150 mm",
        "touchpoints": ["Apply nitrogen-rich fertilizer early", "Ensure consistent moisture", "Use crop rotation to preserve soil health"],
        "image": "img/maize.jpg"
    },
    "sugarcane": {
        "summary": "Sugarcane grows best in tropical and subtropical climates with ample water and nutrients.",
        "ph": "6.0 - 7.5",
        "temperature": "20 - 30°C",
        "rainfall": "150 - 200 mm",
        "touchpoints": ["Keep soil moisture high", "Use potassium-rich fertilizer", "Maintain clean crop rows"],
        "image": "img/sugarcane.jpg"
    },
    "banana": {
        "summary": "Banana plants are best in humid, tropical climates with rich organic soil and steady water supply.",
        "ph": "5.5 - 6.5",
        "temperature": "26 - 30°C",
        "rainfall": "160 - 220 mm",
        "touchpoints": ["Ensure deep, regular watering", "Mulch to conserve moisture", "Use balanced fertilizer with magnesium"],
        "image": "img/banana.jpg"
    },
    "orange": {
        "summary": "Orange orchards favor warm climates with good drainage and consistent humidity for sweet, juicy fruit.",
        "ph": "5.5 - 6.5",
        "temperature": "18 - 29°C",
        "rainfall": "100 - 150 mm",
        "touchpoints": ["Protect from frost", "Ensure full sun exposure", "Feed with micronutrient supplements"],
        "image": "img/orange.jpg"
    },
    "apple": {
        "summary": "Apple trees perform well in temperate climates and need well-drained soil with regular chill hours.",
        "ph": "5.8 - 6.8",
        "temperature": "18 - 24°C",
        "rainfall": "75 - 120 mm",
        "touchpoints": ["Prune annually", "Maintain consistent irrigation", "Monitor pests and diseases"],
        "image": "img/apple.jpg"
    },
    "potato": {
        "summary": "Potatoes require cool temperatures, loose soil, and even moisture for healthy tuber formation.",
        "ph": "5.0 - 6.0",
        "temperature": "15 - 20°C",
        "rainfall": "75 - 110 mm",
        "touchpoints": ["Avoid waterlogging", "Hill soil around stems", "Use phosphorus-rich fertilizer"],
        "image": "img/potato.jpg"
    },
    "chickpea": {
        "summary": "Chickpea is a drought-tolerant pulse crop that improves soil nitrogen and suits cooler dry seasons.",
        "ph": "6.0 - 7.0",
        "temperature": "10 - 25°C",
        "rainfall": "35 - 60 mm",
        "touchpoints": ["Plant in well-drained soil", "Use crop rotation with cereals", "Monitor for pests in dry weather"],
        "image": "img/chickpea.jpg"
    },
    "coffee": {
        "summary": "Coffee grows best in shaded, tropical highlands with rich, acidic soil and regular moisture.",
        "ph": "5.0 - 6.5",
        "temperature": "18 - 24°C",
        "rainfall": "120 - 180 mm",
        "touchpoints": ["Provide partial shade", "Mulch for moisture management", "Apply organic matter frequently"],
        "image": "img/coffee.jpg"
    },
    "cabbage": {
        "summary": "Cabbage grows well in cool weather and prefers fertile, moisture-retentive soil.",
        "ph": "6.0 - 7.0",
        "temperature": "10 - 20°C",
        "rainfall": "60 - 100 mm",
        "touchpoints": ["Keep soil evenly moist", "Use balanced nitrogen feeding", "Protect from heat stress"],
        "image": "img/cabbage.jpg"
    },
    "pea": {
        "summary": "Peas prefer cool conditions, well-drained soil, and moderate moisture for strong growth.",
        "ph": "6.0 - 7.5",
        "temperature": "10 - 20°C",
        "rainfall": "50 - 80 mm",
        "touchpoints": ["Avoid waterlogging", "Provide support for climbing varieties", "Rotate with cereals"],
        "image": "img/pea.jpg"
    },
    "lettuce": {
        "summary": "Lettuce thrives in cool climates with regular moisture and rich, loose soil.",
        "ph": "6.0 - 7.0",
        "temperature": "10 - 20°C",
        "rainfall": "70 - 100 mm",
        "touchpoints": ["Maintain consistent watering", "Protect from bolting in heat", "Use shade during hot spells"],
        "image": "img/lettuce.jpg"
    },
    "spinach": {
        "summary": "Spinach grows best in cool weather and benefits from steady moisture and fertile soil.",
        "ph": "6.0 - 7.0",
        "temperature": "10 - 20°C",
        "rainfall": "50 - 80 mm",
        "touchpoints": ["Harvest young leaves regularly", "Keep soil evenly moist", "Avoid high heat exposure"],
        "image": "img/spinach.jpg"
    },
    "rice": {
        "summary": "Rice is a water-loving cereal that performs well in flooded or irrigated lowland fields.",
        "ph": "5.5 - 7.0",
        "temperature": "20 - 35°C",
        "rainfall": "150 - 250 mm",
        "touchpoints": ["Maintain standing water when needed", "Use nutrient-rich flooded soils", "Manage weeds early"],
        "image": "img/rice.jpg"
    },
    "millet": {
        "summary": "Millet is a hardy, drought-tolerant cereal suited to warm climates and poor soils.",
        "ph": "5.5 - 7.0",
        "temperature": "25 - 35°C",
        "rainfall": "40 - 80 mm",
        "touchpoints": ["Use low-input management", "Plant in well-drained soil", "Avoid overwatering"],
        "image": "img/millet.jpg"
    },
    "soybean": {
        "summary": "Soybean grows well in warm climates with fertile soil and moderate rainfall.",
        "ph": "6.0 - 7.0",
        "temperature": "20 - 30°C",
        "rainfall": "60 - 100 mm",
        "touchpoints": ["Ensure good nodulation", "Use crop rotation with cereals", "Monitor for pests"],
        "image": "img/soybean.jpg"
    },
    "tea": {
        "summary": "Tea prefers cool, humid highlands with acidic soil and regular rainfall.",
        "ph": "4.5 - 6.0",
        "temperature": "15 - 25°C",
        "rainfall": "150 - 250 mm",
        "touchpoints": ["Maintain shade and humidity", "Use acidic fertilizer carefully", "Prune regularly"],
        "image": "img/tea.jpg"
    },
    "cotton": {
        "summary": "Cotton needs warm weather, bright sunlight, and well-drained soil for high yield.",
        "ph": "5.8 - 7.0",
        "temperature": "25 - 35°C",
        "rainfall": "50 - 100 mm",
        "touchpoints": ["Provide full sun", "Use timely irrigation", "Monitor pests and boll health"],
        "image": "img/cotton.jpg"
    },
    "wheat": {
        "summary": "Wheat grows best in cool climates with moderate rainfall and fertile loam.",
        "ph": "6.0 - 7.5",
        "temperature": "10 - 25°C",
        "rainfall": "50 - 90 mm",
        "touchpoints": ["Use proper seeding depth", "Apply nitrogen at key stages", "Protect against lodging"],
        "image": "img/wheat.jpg"
    },
    "tomato": {
        "summary": "Tomato crops prefer warm weather, full sun, and rich, well-drained soil.",
        "ph": "6.0 - 7.0",
        "temperature": "20 - 27°C",
        "rainfall": "80 - 120 mm",
        "touchpoints": ["Support the plants", "Mulch to conserve moisture", "Watch for fungal disease"],
        "image": "img/tomato.jpg"
    },
    "barley": {
        "summary": "Barley is adapted to cool climates and tolerates drier conditions better than many cereals.",
        "ph": "6.0 - 7.5",
        "temperature": "10 - 20°C",
        "rainfall": "40 - 80 mm",
        "touchpoints": ["Use well-drained fields", "Apply balanced nutrients", "Avoid excess moisture"],
        "image": "img/barley.jpg"
    },
    "grapes": {
        "summary": "Grapes prefer sunny, warm climates with well-drained soil and careful irrigation.",
        "ph": "6.0 - 7.0",
        "temperature": "15 - 30°C",
        "rainfall": "60 - 100 mm",
        "touchpoints": ["Provide full sun", "Prune annually", "Use trellising and canopy management"],
        "image": "img/grapes.jpg"
    },
    "mango": {
        "summary": "Mango trees flourish in tropical climates with warm temperatures and good drainage.",
        "ph": "5.5 - 7.5",
        "temperature": "24 - 30°C",
        "rainfall": "75 - 250 mm",
        "touchpoints": ["Ensure good drainage", "Use organic matter in the root zone", "Protect young trees from cold"],
        "image": "img/mango.jpg"
    }
}

DEFAULT_CROP_INFO = {
    "summary": "The hero image will appear here once a crop recommendation is generated.",
    "ph": "N/A",
    "temperature": "N/A",
    "rainfall": "N/A",
    "touchpoints": ["Use local agronomy advice", "Monitor soil and climate closely", "Follow balanced nutrient management"],
    "image": "img/default_crop.jpg"
}

DEFAULT_HERO_IMAGE = "img/default_hero.jpg"


def get_crop_insights(crop_name):
    return CROP_GUIDE.get(crop_name.lower(), DEFAULT_CROP_INFO)


def resolve_crop_image(crop_name):
    crop_info = get_crop_insights(crop_name)
    candidate_paths = [
        crop_info.get("image", DEFAULT_CROP_INFO["image"]),
        DEFAULT_CROP_INFO["image"],
        DEFAULT_HERO_IMAGE,
        "img/default_crop.jpg",
        "img/default_hero.jpg",
    ]

    for image_path in candidate_paths:
        static_path = os.path.join(app.static_folder, image_path.lstrip("/"))
        if os.path.exists(static_path):
            return image_path

    return DEFAULT_CROP_INFO["image"]


def train_and_save_model():
    data_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    df = pd.read_csv(data_path).dropna()
    X = df[FEATURE_COLUMNS]
    y = df["crop"]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler


def ensure_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_and_save_model()


def load_model(path):
    ensure_model_artifacts()
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_input(values, scaler):
    df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    df_scaled = scaler.transform(df)
    return df_scaled


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        hero_image=DEFAULT_HERO_IMAGE,
        crop_image=DEFAULT_HERO_IMAGE,
        hero_crop=None,
        crop_summary=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = {col: float(request.form.get(col, 0)) for col in FEATURE_COLUMNS}
    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values for all fields.",
            hero_image=DEFAULT_HERO_IMAGE,
            crop_image=DEFAULT_HERO_IMAGE,
        )

    if any(value < 0 for value in user_input.values()):
        return render_template(
            "index.html",
            error="Please enter non-negative values for the input parameters.",
            hero_image=DEFAULT_HERO_IMAGE,
            crop_image=DEFAULT_HERO_IMAGE,
        )

    model = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    features = preprocess_input(user_input, scaler)
    prediction = model.predict(features)[0]
    crop_info = get_crop_insights(prediction)
    crop_image = resolve_crop_image(prediction)

    return render_template(
        "index.html",
        result=prediction,
        inputs=user_input,
        hero_image=crop_image,
        hero_crop=prediction,
        crop_image=crop_image,
        crop_summary=crop_info["summary"],
        crop_ph=crop_info["ph"],
        crop_temperature=crop_info["temperature"],
        crop_rainfall=crop_info["rainfall"],
        crop_touchpoints=crop_info["touchpoints"],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
