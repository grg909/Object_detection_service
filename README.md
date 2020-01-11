# Multi-labels classification transfer learning pipline <!-- omit in toc --> 

This is a Resnet18 based multi-labels classification model trained on cocostuff dataset

## Setup

### Requirements

* Python 3.6+
* Pipenv environement

1. Activate exited environment ( for creating new pipenv virtual environment, add --python 3.6）

```sh
$ pipenv shell
```

if found exited environment, skip 2, 3 steps

2. Install required package

```sh
$ pip install -r requirements.txt
```

3. Download pretrained model

```
https://drive.google.com/open?id=1Gj8bZkfU9KKNFnzFeLDJuaJcu3ZWsGSV
```

Download and put it inside the project directory

## Structure

==============================================================

- checkpoints/
- newdata/
    - images/
    - labels/
    - select.pkl
- pycocotools/
- static/
- temp/
- templates/
- app.py
- configs.py
- models.py
- engine.py
- dataloader.py
- utils.py
- main_coco.py

==============================================================

- checkpoints/ : Save the best performing model on the validation dataset
- newdata/ : Save new images and corresponding labels marked on the web platform. label format is .pkl
- pycocotools/ : Cocostuff dataset python api. ref: https://github.com/cocodataset/cocoapi
- static/ : Save web frontend static resource
- temp/ : Save recommend labels temporarily (*need timing script cleanup)
- templates/ : Web frontend template htmls
- app.py : Flask main app to run
- configs.py : Some configurations of flask web service
- model.py : Resnet18 model class
- engine.py : Train function
- dataloader.py: Dateset class, data loader functions
- utils.py : Some useful tool functions for model training
- main.py : main app function to start train
- select.pkl : The list of total labels that used in model

> Pretrained model's labels: (select.pkl)

'sea', 'ground-other', 'platform', 'building-other', 'branch', 'grass', 'hill', 'fog', 'snow', 'leaves', 'river', 'road', 'bush', 'tree', 'bridge', 'flower', 'railroad', 'pavement', 'mountain', 'playingfield', 'house', 'boat', 'fire hydrant', 'person', 'motorcycle', 'stop sign', 'bicycle', 'bench', 'traffic light', 'bus', 'parking meter', 'train', 'truck', 'car'

## Usage

### Web annotation platform

A web annotation platform can submit new images with annotation, and create new dataset

```
Usage: gunicorn -c gunicorn_config.py app:app
```

(To restart the server, execute "$ pipenv shell" and then the above command)

Then check http://ip:7777

### web api server

 1. /imgup
    
    > using for upload img files
    
    ##### request:
    
    -  args
    
        | args | nullable | type | remark |
        |:------:|:-----:|:-----:|:------:|
        | id_name | false | str    | 图片id，如‘1’,'a_1'                                    |

    -  multipart-encoded file
    
        Read file with 'rb' mode
        >files = {'file': open('report.xls', 'rb')}
        
        Send a post request
        >requests.post(url, files=files, data={"id_name":"1"})
        
     -  request the sample:
        ```
        POST /imgup HTTP/1.1
        Host: 192.168.23.115:7777
        file: files
        {"id_name":"1"}
        ```
    
    ##### Return:
    
    success: 200
    
    failed: 400
    
 2. /imginfo
    
    > return the GPS and labels info of img
    
    ##### request:
    
    -  args
    
        | args | nullable | type | remark |
        |:------:|:-----:|:-----:|:------:|
        | id_name | false | str    | 图片id，如‘1’,'a_1'                                    |
   
    
    ##### Return:
    
    - args
    
        | args | type | remark |
        |:------:|:-----:|:-----:|
        | image_id   | str      | 上传时设定的图片id_name |
        | predict_tags     | list      | 预测图片中出现的labels |
        | data_information | str      | 图片中的时间信息 |
        | GPS_information | dict      | 图片中的位置信息 |
        
        全部支持的label见上Pretrained model's labels
        
     - GPS_information

        | args | type | remark |
        |:------:|:-----:|:-----:|
        | GPSAltitude | str      | GPS高度 |
        | GPSAltitudeRef        | str      | GPS高度标识 |
        | GPSLatitude        | float      | GPS纬度 |
        | GPSLatitudeRef        | str      | GPS纬度标识 |
        | GPSLongitude      | float      | GPS经度         |
        | GPSLongitudeRef    | str      | GPS经度标识  |
            
    - example
     
        ```
        {
            "image_id": "1",
            "predict_tags": [
                "building-other"
                "tree",
                "mountain",
                "train"
            ],
            "data_information": "2019:08:03 14:41:10",
            "GPS_information": {
                "GPSAltitude":"0",
                "GPSAltitudeRef":"1",
                "GPSLatitude":28.835149765,
                "GPSLatitudeRef":"N",
                "GPSLongitude":106.93543243388889,
                "GPSLongitudeRef":"E"
            }
        }
        ```
    
### Transfer learning function

```
Usage: python main_coco.py [OPTIONS]

  Transfer learning: Train the output fc layer of pretrained model based on new dataset

Options:
  -h, --help                               show this help message and exit
  --workers N                              number of data loading workers (default: 4)
  --epochs N                               number of total epochs to run
  --epoch_step EPOCH_STEP                  number of epochs to change learning rate
  --device_ids DEVICE_IDS                  ids of GPU
  --start-epoch N                          manual epoch number (useful on restarts)
  -b N, --batch-size N                     mini-batch size (default: 256)
  --lr LR, --learning-rate LR              initial learning rate
  --lrp LR, --learning-rate-pretrained LR  learning rate for pre-trained layers
  --momentum M                             momentum
  --weight-decay W, --wd W                 weight decay (default: 1e-4)
  --print-freq N, -p N                     print frequency (default: 10)
  --resume PATH                            path to latest checkpoint (default: none)
  --result PATH                            path to save prediction results (default: none)
```
> Example

python main.py -b 4 --resume '../checkpoints/checkpint_0820.tar' --epochs 10 --result '../output/coco_0820.csv'



