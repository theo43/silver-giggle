from fastapi import FastAPI
import tensorflow as tf
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parent
MODEL_PATH = BASE_PATH / "models" / "one_step_model.keras"

app = FastAPI(
    title='Shakespeare Model API',
    description='API to make predictions using Shakespeare model',
)

# Load the OneStep model
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def predict(text: str):
    predictions = model.generate_one_sentence(text)

    return {"predictions": predictions}
