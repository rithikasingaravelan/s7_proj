from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Load the trained CNN model
model = load_model('plantdiseasenaivecnn8epoch.h5')

# Dictionary mapping disease classes to preventive measures
preventive_measures = {
    'Citrus canker': '1. Use disease-resistant plant varieties. 2. Apply copper-based fungicides.',
    'Mancha Graxa': '1. Rotate crops to reduce disease pressure. 2. Remove and destroy infected plants.',
    'Tomato mosaic virus': '1. Use virus-free seeds. 2. Control insect vectors. 3. Remove and destroy infected plants.',
    'Tomato Yellow Leaf Curl Virus': '1. Use virus-resistant plant varieties. 2. Control whitefly populations.'
}


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
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            prediction = predict_disease(image_path)

            # Assuming you have a list of classes corresponding to the disease labels
            classes = ['Citrus canker', 'Mancha Graxa', 'Tomato mosaic virus',
                       'Tomato Yellow Leaf Curl Virus']  # Update this list with your classes
            predicted_class = classes[np.argmax(prediction)]

            # Get preventive measures for the predicted class
            measures = preventive_measures.get(predicted_class, 'No specific preventive measures available.')

            # Return the prediction and preventive measures as JSON data
            return jsonify(prediction=predicted_class, preventive_measures=measures)

    # Return an empty JSON response if no prediction can be made
    return jsonify(prediction=None, preventive_measures=None)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
