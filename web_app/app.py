from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import threading
import queue
import time
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_app/static/uploads'

confidence_interval = 0.2

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def b64encode(data):
    return base64.b64encode(data).decode('utf-8')

app.jinja_env.filters['b64encode'] = b64encode

webcam_active = False
webcam_thread = None
frame_queue = queue.Queue(maxsize=1)  # Limit queue size to prevent memory buildup

# Load model once at startup
model = YOLO("web_app/static/model.pt").to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Uploading file")
    global webcam_active
    global webcam_thread

    if webcam_active:
        webcam_active = False
        webcam_thread.join()

    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Ensure consistent image processing
        img = img.convert('RGB')  # Convert to RGB mode
        img_array = np.array(img)

        # Resize image to a consistent size
        img_array = cv2.resize(img_array, (640, 480))

        try:
            results = model([img_array], conf=confidence_interval)
            try:
                sign = results[0].names[int(results[0].boxes.cls[0].int())]
                print(sign) 
            except:
                sign = "No Sign Detected!"
        except Exception as e:
            print(f"Inference error: {e}")
            sign = "Detection Error"

        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)
        img_bytes = img_io.getvalue()

        return render_template('result.html', original_image=img_bytes, sign=sign)
    


def generate_frames():
    global webcam_active
    cap = cv2.VideoCapture(0)
    
    # Reduce resolution to improve processing speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_process_time = time.time()
    process_interval = 0.1  # Process every 0.1 seconds

    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            # Non-blocking inference
            try:
                results = model.track(frame, persist=True, conf=confidence_interval)
                annotated_frame = results[0].plot()
                
                # Try to put latest frame in queue, replacing old frame if queue is full
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                
                frame_queue.put_nowait(annotated_frame)
                last_process_time = current_time
            except Exception as e:
                print(f"Processing error: {e}")

@app.route('/video_feed')
def video_feed():
    def generate():
        while webcam_active:
            try:
                frame = frame_queue.get(timeout=1)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                continue
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
def start_webcam():
    global webcam_active
    global webcam_thread

    if not webcam_active:
        webcam_active = True
        webcam_thread = threading.Thread(target=generate_frames, daemon=True)
        webcam_thread.start()
    return "Webcam started"

@app.route('/stop_webcam')
def stop_webcam():
    global webcam_active
    global webcam_thread

    if webcam_active:
        webcam_active = False
        webcam_thread.join()
    return "Webcam stopped"

if __name__ == '__main__':
    app.run(debug=True, port=5000)