# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    flask main app
"""

from flask import Flask, request, render_template, jsonify, session, escape, abort, redirect
from configs import Configs
from flask_dropzone import Dropzone
from deeplab import DeeplabPytorch
from label_json import label_json
from collections import defaultdict
import pickle
from flask_jsglue import JSGlue
import os
from glob import glob
from time import time
import hashlib
from random import random
from annotator.python.generate_data_json import gen_annotation_json


app = Flask(__name__)
app.config.from_object(Configs)
dropzone = Dropzone(app)
jsglue = JSGlue(app)


def clean_own(id_pattern):

    r = glob(id_pattern)
    for i in r:
        os.remove(i)


@app.route('/', methods=['GET', 'POST'])
def uploads():
    id_gen = random()
    if 'uid' not in session:
        user_id = str(id_gen).split('.')[1]
        session['uid'] = user_id

    if request.method == 'POST':
        user_id = escape(session.get('uid'))
        file = request.files.get('file')
        name, ext = os.path.splitext(file.filename)
        id_name = user_id + '_' + hashlib.md5((name + str(time())).encode('UTF-8')).hexdigest()[:10]
        new_name = id_name + ext
        # try:
        file.save(os.path.join('annotator/data/images/', new_name))
        # labelmap = dp.single(file.read(), id_name)
        # pickle.dump(labelmap, open('data/temp/{}.pkl'.format(id_name), 'wb'))
        return jsonify(success=True), 200
        # except:
        #     return '上传失败了！', 400

    return render_template('uploads.html')


@app.route('/results', methods=['GET'])
def results():
    gen_annotation_json()

    return redirect('/annotator')


# dp = DeeplabPytorch(config_path = 'configs/cocostuff164k.yaml',
#                     model_path = 'data/models/coco/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')

# def mark(labelmap_cache):
#
#     labelmap = pickle.load(open('data/temp/{}.pkl'.format(labelmap_cache), 'rb'))
#     result_raw = label_json(labelmap)
#     result_clean = {key: round(value, 5) for key, value in result_raw.items() if value > 0}
#     result = dict(sorted(
#        result_clean.items(),
#        key=lambda t: t[1],
#        reverse=True))
#
#     return result


# deeplab_pytorch predict
# @app.route('/results', methods=['GET'])
# def results():
#
#     user_id = escape(session['uid'])
#     results = []
#     for file in os.listdir('data/temp/'):
#         if file.endswith('.pkl') and file.startswith(user_id):
#             id_name = os.path.splitext(file)[0]
#             results.append((id_name, mark(id_name)))
#             os.remove('data/temp/'+file)
#     session.pop('uid')
#     return render_template('result.html', results=results)



