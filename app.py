# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    flask main app
"""

from flask import Flask, redirect, url_for, request, render_template, jsonify
from pypinyin import lazy_pinyin
from werkzeug.utils import secure_filename
import json
from configs import Configs
from flask_dropzone import Dropzone
from flask_bootstrap import Bootstrap
import os
import time


app = Flask(__name__)
app.config.from_object(Configs)
dropzone = Dropzone(app)
bootstrap = Bootstrap(app)

# is file allowed to be uploaded?
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def uploads():
    if request.method == 'POST':
        file = request.files.get('file')
        filename = secure_filename(file.filename)
        if filename.startswith('.'):
            name, ext = os.path.splitext(file.filename)
            filename = '_'.join(lazy_pinyin(name)) + '.' + ext
            #labelmap = dp.single(file.read(), filename)
            #labelmap = {'ball':0.19231}
            #result_raw = label_json(labelmap)
            #result_clean = {key: round(value, 3) for key, value in result_raw.items() if value > 0}
            #result = dict(sorted(
            #    result_clean.items(),
            #    key=lambda t: t[1],
            #    reverse=True))
        print(filename)
        time.sleep(1)
        return jsonify(success=True)
    return render_template('uploads.html')

