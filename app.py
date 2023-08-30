from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained CNN model
model = load_model('plantdiseasenaivecnn8epoch.h5')


# Function to preprocess the uploaded image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Change the target_size according to your model input size
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image


# Function to perform prediction using the loaded model
def predict_disease(image_path):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return prediction


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle the uploaded image and show the prediction result
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            prediction = predict_disease(image_path)
            # Assuming you have a list of classes corresponding to the disease labels
            classes = ['Citrus canker', 'Mancha Graxa']  # Update this list with your classes
            predicted_class = classes[np.argmax(prediction)]
            return render_template('index.html', prediction=predicted_class, image_file=file.filename)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
