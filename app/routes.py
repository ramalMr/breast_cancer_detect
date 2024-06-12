import os
from flask import render_template, request, redirect, url_for
from app import app
from app.utils import predict_vgg16, predict_resnet50, predict_inception

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            vgg16_pred = predict_vgg16(file_path)
            resnet50_pred = predict_resnet50(file_path)
            inception_pred = predict_inception(file_path)
            
            preds = [vgg16_pred, resnet50_pred, inception_pred]
            avg_pred = sum(preds) / len(preds)
            
            if avg_pred > 0.5:
                result = 'Xərçəng xəstəliyi aşkarlandı.'
            else:
                result = 'Xərçəng xəstəliyi aşkarlanmadı.'
                
            return render_template('result.html', result=result, vgg16_pred=vgg16_pred, 
                                   resnet50_pred=resnet50_pred, inception_pred=inception_pred)
    return render_template('index.html')