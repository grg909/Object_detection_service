from label_json import label_json
from lib.core import DeeplabPytorch
from flask import Flask, Response
import json

app = Flask(__name__)


@app.route('/get_json')
def get_json():

    dp = DeeplabPytorch(config_path='configs/cocostuff164k.yaml',
                        model_path='data/models/coco/deeplabv2_resnet101_msc-cocostuff164k-100000.pth',
                        image_path='IMG_2885.JPG')
    labelmap = dp.single()
    result = label_json(labelmap)
    res = json.dumps(result)

    return Response(response=res, status=200, mimetype="application/json")
