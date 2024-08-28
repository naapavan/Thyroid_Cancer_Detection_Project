from os import access
from flask import Flask, render_template, request, redirect, url_for
from flask_ngrok import run_with_ngrok
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

from keras.models import load_model
import cv2
# import pickle

app = Flask(__name__,static_url_path='/static')



model = load_model('epoch_30.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Load the machine learning model
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

@app.route('/')
def hello_world():
    return render_template('model.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the uploaded image file
#     img_file = request.files['image']

#     # Save the uploaded image temporarily
#     img_path = "temp_image.jpg"
#     img_file.save(img_path)

#     # Load and preprocess the image for prediction
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     # Make a prediction using your model
#     predictions = model.predict(img_array)
#     # predicted_class = np.argmax(predictions,axis=-1)

#     # Render the template with the prediction result
#     return render_template('model.html', predicted_class=predictions)
# ...

@app.route('/predict', methods=['POST'])
def predict():


    # Get the uploaded image file
    img_file = request.files['image']

    # model = load_model('epoch_30.h5')

    # model.compile(loss='categorical_crossentropy',
    #           optimizer='adam',
    #           metrics=['accuracy'])

    class_names=['4A','4B','4C','5','Benign','normal thyroid']


    # Save the uploaded image temporarily

    img_path = "temp_image.jpg"
    img_file.save(img_path)

    # Load and preprocess the image for prediction
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make a prediction using your model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    # Pass the image path and prediction to the template
    return render_template('model.html', predicted_class=predicted_class_name)

# ...


if __name__ == '__main__':
    app.run(debug=True)
