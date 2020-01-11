# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""
    deep learning classifier web serviceapp
"""

from flask import Flask, request, render_template, jsonify, session, escape, abort, redirect, flash, url_for
from configs import Configs
from flask_dropzone import Dropzone
from xxktt_api import XxkttApi
from collections import defaultdict
import pickle
from flask_jsglue import JSGlue
import os
from glob import glob
from time import time
import hashlib
from random import random
import numpy as np
import PIL
from PIL import Image
from exifread import process_file
import re

app = Flask(__name__)
app.config.from_object(Configs)
dropzone = Dropzone(app)
jsglue = JSGlue(app)


dp = XxkttApi(checkpoint_path='../checkpoints/checkpint_0812.tar', label_path='../coco2017/select.pkl')


def latitude_and_longitude_convert_to_decimal_system(*arg):
    """
    经纬度转为小数, param arg:
    :return: 十进制小数
    """
    return float(arg[0]) + ((float(arg[1]) + (float(arg[2].split('/')[0]) / float(arg[2].split('/')[-1]) / 60)) / 60)


def create_thumbnail(file_path, id_name):
    base_width = 180
    img = Image.open(file_path)
    img = img.convert("RGB")
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)
    img.save('static/'+ id_name + '.jpg')


def predit_to_pkl(id, img_path, pkl_path='temp'):
    preds_tag = dp.single(img_path)
    preds_tag = preds_tag[0]
    pickle.dump((img_path, preds_tag), open('{}/{}.pkl'.format(pkl_path, id), 'wb'))


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
        id_name = user_id + '_' + hashlib.md5((name + str(time())).encode('UTF-8')).hexdigest()[-10:]
        new_name = id_name + ext
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)

        try:
            file.save(file_path)
            create_thumbnail(file_path, id_name)
            predit_to_pkl(id_name, file_path, 'temp')
            return jsonify(success=True), 200
        except:
            return '上传失败了！', 400

    return render_template('uploads.html')


@app.route('/results', methods=['GET'])
def results():

    user_id = escape(session['uid'])
    results = []
    for file in os.listdir('temp/'):
        if file.endswith('.pkl') and file.startswith(user_id):
            id_name = os.path.splitext(file)[0]
            print(id_name)
            _, preds_tag = pickle.load(open('temp/{}.pkl'.format(id_name), 'rb'))
            results.append((id_name, preds_tag))
    return render_template('result.html', results=results)


@app.route('/labels/<id_name>', methods=['POST'])
def labels(id_name):

    file_path, preds_tag = pickle.load(open('temp/{}.pkl'.format(id_name), 'rb'))
    anno_tag = {}
    for i in preds_tag:
        if request.form.get(i):
            anno_tag[i] = 1
    print(anno_tag)
    pickle.dump((file_path, anno_tag), open('newdata/labels/{}.pkl'.format(id_name), 'wb'))
    os.remove('temp/{}.pkl'.format(id_name))

    flash('该图片标注信息已保存！')  # 显示成功创建的提示

    return redirect(url_for('results'))


@app.route('/logout')
def logout():

    user_id = escape(session['uid'])
    for file in os.listdir('temp/'):
        if file.startswith(user_id):
            os.remove('temp/'+file)
    session.pop('uid')

    return redirect(url_for('uploads'))


@app.route('/imginfo/<id_name>')
def imginfo(id_name):

    file_path, preds_tag = pickle.load(open('temp/{}.pkl'.format(id_name), 'rb'))

    with open(file_path, 'rb') as fb:
        tags = process_file(fb)
        GPS = {}
        date = ''
        for tag, value in tags.items():
            if re.match('GPS GPSLatitudeRef', tag):
                GPS['GPSLatitudeRef'] = str(value)
            elif re.match('GPS GPSLongitudeRef', tag):
                GPS['GPSLongitudeRef'] = str(value)
            elif re.match('GPS GPSAltitudeRef', tag):
                GPS['GPSAltitudeRef'] = str(value)
            elif re.match('GPS GPSLatitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLatitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLatitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSLongitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLongitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLongitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSAltitude', tag):
                GPS['GPSAltitude'] = str(value)
            elif re.match('.*Date.*', tag):
                date = str(value)

    return jsonify({'image_id': id_name, "predict_tags": preds_tag, 'date_information': date, 'GPS_information': GPS})




