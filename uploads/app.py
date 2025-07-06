from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input 
import numpy as np
import tensorflow as tf
from PIL import Image
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

model = tf.keras.models.load_model('Vgg16_97.h5')

labels = ["default_product", "good_product"]

@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template("index.html")

@app.route('/about')
def about():
    """Renders the about.html page."""
    return render_template("about.html")

@app.route('/contact')
def contact():
    """Renders the contact.html page."""
    return render_template("contact.html")

@app.route('/predict')
def predict():
    """Renders the inner-page.html, presumably where image upload form is located."""
    return render_template("inner-page.html")

@app.route('/output', methods=['POST'])
def output():
    """
    Handles image upload, preprocessing, model prediction,
    and renders the result on 'portfolio-details.html'.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("error.html", message="No file included in the request.")

        f = request.files['file']

        if f.filename == '':
            return render_template("error.html", message="No file selected.")

        upload_dir = os.path.join(app.root_path, 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        filepath = os.path.join(upload_dir, f.filename)
        f.save(filepath)

        try:
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) 
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)

            if preds < 0.5:
                res = 'faulty product'
            else:
                res = 'good product'

            return render_template("portfolio-details.html", predict=res)

        except Exception as e:
            return render_template("error.html", message=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=8000)
