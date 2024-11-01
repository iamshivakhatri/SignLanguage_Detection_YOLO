# app.py
import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import torch

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Load your model

# Create the uploads directory if it doesn't exist
os.makedirs('static/uploads', exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filepath = os.path.join('static/uploads', file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = cv2.imread(filepath)
    results = model(img)  # Perform inference

    # Annotate the image with detections
    annotated_img = results.render()[0]

    # Save annotated image
    annotated_filepath = os.path.join('static/uploads', 'annotated_' + file.filename)
    cv2.imwrite(annotated_filepath, annotated_img)

    return redirect(url_for('show_result', filename='annotated_' + file.filename))


@app.route('/result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
