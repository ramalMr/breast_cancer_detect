import os
from google_drive_downloader import GoogleDriveDownloader as gdd

model_ids = {
    'vgg16': '1-0dzGxFGO8vTFQdK5WVbEPZJFb5Qxu1q',
    'resnet50': '1-1x7JOFjqJQk2NSbCDkX0za7JC0VLGr2',
    'inception': '1-2uV5Edzh1qKmBi7WFMKDuvuf4ov4iKt'
}

def download_model(model_name, file_id):
    model_path = f'models/{model_name}_model.h5'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        gdd.download_file_from_google_drive(file_id=file_id, dest_path=model_path, unzip=False)

if __name__ == '__main__':
    for model_name, file_id in model_ids.items():
        download_model(model_name, file_id)