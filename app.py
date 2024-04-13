from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)
socketio = SocketIO(app)

# Load the YOLOv8 model with your custom weights
weights_path = 'runs/detect/yolov8m_v8_50bigfood150/weights/best.pt'
model = YOLO(weights_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the nutritional information, colors, and class types
nutritional_info = {
    # Your nutritional info dictionary
}
green_box_color = (0, 255, 0)
red_box_color = (0, 0, 255)
text_color = (255, 255, 255)
green_classes = ['dal_makhni', 'kadai_paneer', 'chapati']
red_classes = ['pizza', 'samosa', 'burger']

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    # Decode the image from base64
    image_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference using YOLO
    results = model(frame)

    # Process detections and draw bounding boxes and text
    for det in results[0].boxes:
        if det.xyxy.numel() >= 4:
            coords = det.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(coord) for coord in coords[:4]]
            conf = det.conf.item()
            class_index = det.cls.item()
            class_name = model.names[int(class_index)]

            # Choose color based on class
            box_color = green_box_color if class_name in green_classes else red_box_color

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Prepare and put the text
            text = f'{class_name}: {conf:.2f}'
            cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Add nutritional info if class_name is in the dictionary
            if class_name in nutritional_info:
                info_text = nutritional_info[class_name]
                cv2.putText(frame, info_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Encode the modified frame
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    emit('response', {'image': f'data:image/jpeg;base64,{frame_data}'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
