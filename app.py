import pickle
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from flask import Flask, render_template, request
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
model1 = pickle.load(open('model.pkl', 'rb'))

# Load YOLOv5 model
model_path = 'best.pt'  # path to the YOLOv5 weights file

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('input_form.html')


@app.route('/process_input', methods=['POST'])
def process_input():
    if request.method == 'POST':
        file = request.files.get('file')
        file.save("Input_image/input.jpg")
        print("hello")
        try:
            print("1")
            # Load the image
            image_path = 'Input_image/input.jpg'  # Replace with the path to your image
            print("2")
            image = Image.open(image_path)

            # Load the YOLOv7 model
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'C:/Users/cyril/PycharmProjects/Tarp/best.pt')
            print("3")

            output = model(image)
            print("5")
            print(output)
            print("6")
            # Save image with bounding boxes drawn
            output.save("static/")

            N = random.uniform(0.0, 200.0)
            P = random.uniform(0.0, 200.0)
            K = random.uniform(0.0, 200.0)
            temperature = random.uniform(0.0, 100.0)
            humidity = random.uniform(0.0, 200.0)
            ph = random.uniform(0.0, 14.0)
            rainfall = random.uniform(0.0, 2000.0)
            price = random.uniform(0.0, 10000.0)

            input1 = np.array([N, P, K, temperature, humidity, ph, rainfall, price]).reshape(1, -1)
            # Do something with the input values here

            dataset = pd.read_csv("C:/Users/cyril/PycharmProjects/Tarp/indiancrop_dataset.csv")

            df1 = pd.DataFrame(dataset)
            y = pd.DataFrame(df1['label'])
            label_encoder.fit_transform(y)

            output1 = model1.predict(input1)
            lab = label_encoder.inverse_transform(output1)
            return render_template('predicted.html', result=lab)

        except Exception as e:
            return str(e)

    else:
        # Print an error message
        return "Error uploading file."


if __name__ == "__main__":
    app.run(debug=True)
