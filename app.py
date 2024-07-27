from PIL import Image
import numpy as np
from flask import Flask, request
import io
from libgen_api import LibgenSearch

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
@app.route('/search', methods=['GET'])
def search_book():
    # Get the book title from the request arguments
    book_title = request.args.get('q')

    # Perform the search operation (replace this with actual search logic)
    search_result = search_in_genesis(book_title)

    # Return the search result
    return search_result, 200

def search_in_genesis(book_title):
    # Placeholder function for searching a book in the Genesis library
    # Replace this with actual search logic
    print(book_title)
    s = LibgenSearch()
    results = s.search_title(book_title)
    print(results)
    return results
if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0', port= 5000)