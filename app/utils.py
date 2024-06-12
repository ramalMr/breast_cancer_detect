import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from app.models import vgg16_model, resnet50_model, inception_model

def predict_vgg16(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
    prediction = vgg16_model.predict(img_preprocessed)
    return prediction[0][0]

def predict_resnet50(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)
    prediction = resnet50_model.predict(img_preprocessed)
    return prediction[0][0]

def predict_inception(file):
    img = image.load_img(file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.inception_v3.preprocess_input(img_batch)
    prediction = inception_model.predict(img_preprocessed)
    return prediction[0][0]