import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Configs:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    UPLOADED_PHOTOS_DEST  = os.getcwd() + '/static'
    DROPZONE_ALLOWED_FILE_TYPE='image'
    DROPZONE_MAX_FILE_SIZE=10
    DROPZONE_MAX_FILES=30
