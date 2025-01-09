from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the model
model = load_model('model/chest_xray_blur_model.h5')

# Print the input shape of the model to determine if it's RGB or Grayscale
print(model.input_shape)  # This will print the model's input shape

@app.route('/')
def home():
    return "Welcome to the Flask app!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(150, 150), color_mode='grayscale')  # Load as grayscale
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img_array)
    
        # Convert prediction to result
        result = "Pneumonia" if prediction > 0.5 else "Normal"

        return jsonify({"prediction": result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
