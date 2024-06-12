import os
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def create_model(model_name):
    model_path = f'models/{model_name}_model.h5'
    
    if model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'inception':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
    
    return model

vgg16_model = create_model('vgg16')
resnet50_model = create_model('resnet50')
inception_model = create_model('inception')