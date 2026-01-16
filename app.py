from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model("model/plant_disease_model.h5")

classes = {
    0: ("Healthy Plant", "No disease detected.", "No treatment needed.", "Maintain good practices."),
    1: ("Early Blight", "Fungal disease causing leaf spots.", "Use fungicides.", "Avoid leaf wetness."),
    2: ("Late Blight", "Severe fungal infection.", "Apply copper-based fungicides.", "Remove infected plants."),
    3: ("Leaf Rust", "Rust-colored fungal spots.", "Sulfur fungicide recommended.", "Ensure good airflow.")
}

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        result = classes[predicted_class]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
