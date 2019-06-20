# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    flask main app
"""

from label_json import label_json
from deeplab import DeeplabPytorch
from flask import Flask, Response, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import json


UPLOAD_FOLDER = 'data/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
dp = DeeplabPytorch(
    config_path='configs/cocostuff164k.yaml',
    model_path='data/models/coco/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')

# main route
@app.route('/')
def index():
    return render_template('uploads.html')


# is file allowed to be uploaded?
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# file upload route
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        labelmap = dp.single(file.read(), filename)
        result_raw = label_json(labelmap)
        result_clean = {key: round(value, 3) for key, value in result_raw.items() if value > 0}
        result = dict(sorted(
            result_clean.items(),
            key=lambda t: t[1],
            reverse=True))

        return render_template('result.html', result=result, filename=filename)


@app.route('/get_json')
def get_json():

    labelmap = dp.single(image_path='IMG_2885.JPG')
    result = label_json(labelmap)
    res = json.dumps(result)

    return Response(response=res, status=200, mimetype="application/json")
