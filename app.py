import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# The Flask constructor takes the name of the current module as argument
app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

# Define your 6 fruit categories in the EXACT order of your dataset folders
class_names = ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']

@app.route('/')
def home():
    # Renders the landing page
    return render_template('index.html')

@app.route('/inspect')
def inspect():
    # Renders the inspection/prediction page when "Inspect" is clicked
    return render_template('inner.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    
    # Ensure the static folder exists to store and display images
    if not os.path.exists('static'):
        os.makedirs('static')
        
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Image preprocessing to match the training target size
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Convert to array for prediction

    # Make the prediction using the loaded model
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]

    # Render result back to inner.html as per documentation requirements
    return render_template('inner.html', prediction=result, image_path=file.filename)

if __name__ == "__main__":
    app.run(debug=True)