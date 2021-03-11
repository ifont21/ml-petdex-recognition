from flask import Flask, request
from utils import process_image, ml_model

app = Flask(__name__)


@app.route('/api/recognition', methods=['POST'])
def recognition():
    print("called ...", request.json)
    filename = request.json['filename']
    loaded_model = ml_model.load_model(
        './models/20210303-18591614797989-full-image-set-mobilenetv2-Adam.h5')
    print('loaded_model', loaded_model)
    img = process_image.process_image(f'./pictures/{filename}')
    predicted_label = ml_model.get_prediction_label(loaded_model, img)

    return {
        "label": predicted_label
    }
