# -*- coding: UTF-8 -*-

# @Date    : 2019/6/18
# @Author  : WANG JINGE
# @Email   : wang.j.au@m.titech.ac.jp
# @Language: python 3.6
"""

"""

import numpy as np
import pandas as pd
import json


def pd_parse(data):

    raw = data.to_dict()
    res = {}
    for key, value in raw.items():
        res[key] = value[0]
    return res


def label_json(labelmap):


    label_count = 182
    confusion = np.zeros((label_count, label_count))

    labels = open('labels.txt', 'r')

    X = pd.read_csv(labels, sep="\t", header=None)
    X = X[1:]
    X.columns = ['number', 'label']
    X['number'] = X['number'].astype('int')
    X['label'] = X['label'].astype('str')

    X_labels = X['label']
    X_labels.as_matrix()

    id = np.array(['ImageID', 'PointID'])
    header = np.concatenate((id, X_labels), axis=0)

    data = labelmap
    output = pd.DataFrame(columns=header)
    data = np.squeeze(data)
    data = np.transpose(data)
    data = data[:513, :513]
    cat_ids = range(0, 182)
    valid = np.reshape(np.in1d(data, cat_ids), data.shape)
    valid_pred = data[valid].astype(int)
    height = data.shape[0]
    width = data.shape[1]
    size = height * width
    image_df = pd.DataFrame(valid_pred, columns=['class'])
    df = pd.DataFrame({'count': image_df.groupby(['class']).size()}).reset_index()
    df['labelnum'] = df['class'] + 1
    df_label = pd.merge(df, X, how='right', left_on=['labelnum'], right_on=['number'])
    df_label['ratio'] = df_label['count'] / size
    df_ratio = df_label.transpose()
    df_ratio.columns = df_ratio.loc['label']
    df_final = df_ratio.loc['ratio']
    df_final = pd.DataFrame(df_final)
    df_final = df_final.transpose()
    df_final['ImageID'] = 0
    df_final['PointID'] = 0
    df_final = df_final.fillna(0)
    df_final.reset_index(inplace=True)  # Resets the index, makes factor a column
    df_final.drop("index", axis=1, inplace=True)

    output = pd.concat([output, df_final])
    result = pd_parse(output)

    return result