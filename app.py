from PIL import Image
import numpy as np
from flask import Flask, request
import io

from src.test import model_predicts

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400

    if file:
        # Read the file into an image
        image = Image.open(file)
        # Convert the image into a numpy array
        image_array = np.array(image)
        # Now you can use image_array as you wish
        print(image_array.shape)
        words = model_predicts(image_array)
        return words, 200

    return 'Unexpected error occurred.', 500

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0')