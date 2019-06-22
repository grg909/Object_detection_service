# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    flask main app
"""

from flask import Flask, request, render_template, jsonify
from configs import Configs
from flask_dropzone import Dropzone
from flask_bootstrap import Bootstrap
from deeplab import DeeplabPytorch
from label_json import label_json
from collections import defaultdict
import pickle
import os
from glob import glob
from time import time
import hashlib
import ipaddress


app = Flask(__name__)
app.config.from_object(Configs)
dropzone = Dropzone(app)
bootstrap = Bootstrap(app)
dp = DeeplabPytorch(config_path = 'configs/cocostuff164k.yaml',
                    model_path = 'data/models/coco/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')


user_dict = defaultdict(lambda: 0)


def clean_own(id_pattern):

    r = glob(id_pattern)
    for i in r:
        os.remove(i)


@app.route('/', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        user_ip = int(ipaddress.IPv4Address(request.remote_addr))
        if not user_dict[user_ip]:
            clean_own('static/{}*.jpg'.format(user_ip))
            user_dict[user_ip] += 1
        file = request.files.get('file')
        name, ext = os.path.splitext(file.filename)
        id_name = str(user_ip) + hashlib.md5((name + str(time())).encode('UTF-8')).hexdigest()[:5]

        labelmap = dp.single(file.read(), id_name)
        pickle.dump(labelmap, open('data/temp/{}.pkl'.format(id_name), 'wb'))
        return jsonify(success=True), 200

    return render_template('uploads.html')


def mark(labelmap_cache):

    labelmap = pickle.load(open('data/temp/{}.pkl'.format(labelmap_cache), 'rb'))
    result_raw = label_json(labelmap)
    result_clean = {key: round(value, 5) for key, value in result_raw.items() if value > 0}
    result = dict(sorted(
       result_clean.items(),
       key=lambda t: t[1],
       reverse=True))

    return result


@app.route('/results', methods=['GET'])
def results():

    results = []
    user_ip = int(ipaddress.IPv4Address(request.remote_addr))
    for file in os.listdir('data/temp/'):
        if file.endswith('.pkl') and file.startswith(str(user_ip)):
            id_name = os.path.splitext(file)[0]
            results.append((id_name, mark(id_name)))
            os.remove('data/temp/'+file)

    return render_template('result.html', results=results)



