from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model("CNN.h5")

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
from PIL import Image

# ...

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    file = request.files['image']

    # Load and preprocess the image
    img = Image.open(file)
    img = img.resize((180, 180))  # Resize the image to match the expected input shape
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Perform the prediction
    result = model.predict(img)
    probabilities = tf.nn.softmax(result).numpy()[0]

    classes = ['Normal', 'Benign', 'Malignant']
    predicted_class = classes[np.argmax(probabilities)]
    probabilities = {classes[i]: probabilities[i] for i in range(len(classes))}

    # Render the prediction template with the results
    return render_template('prediction.html', class_label=predicted_class, probabilities=probabilities)

# ...


if __name__ == '__main__':
    app.run(debug=True)
