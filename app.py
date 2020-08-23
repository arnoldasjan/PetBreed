from flask import render_template, jsonify
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from prediction import get_prediction
from PIL import Image
#
#
UPLOAD_FOLDER = "/Users/arnoldasjanuska/PycharmProjects/PetBreed/static/img/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
#
# app = Flask(__name__)
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
#
#
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
#
# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             img_bytes = file.read()
#             class_id, class_name = get_prediction(image_bytes=img_bytes)
#             return jsonify({'class_id': class_id, 'class_name': class_name})
#     return render_template('homepage.html')
#
#
# if __name__ == '__main__':
#     app.run()

import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import requests


app = Flask(__name__)
imagenet_class_index = json.load(open('static/imagenet_class_index.json'))
model = models.resnet34(pretrained=True)
model.eval()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resp = requests.post("http://127.0.0.1:5000/predict",
                                 files={"file": open(
                                     os.path.join(app.config['UPLOAD_FOLDER'], filename),
                                     'rb')})
            json_data = resp.json()
            return render_template('answer.html', prediction=json_data['class_name'], img_link=os.path.join("static/img/uploads/", filename))
    return render_template('homepage.html')


if __name__ == '__main__':
    app.run()