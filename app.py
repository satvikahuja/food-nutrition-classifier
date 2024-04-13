from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import base64
import io

app = Flask(__name__)
socketio = SocketIO(app)

# Load the YOLOv8 model with your custom weights
weights_path = 'runs/detect/yolov8m_v8_50bigfood150/weights/best.pt'
model = YOLO(weights_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

nutritional_info = {
    'burger': 'Calories: 540, Protein: 34g',
    'chapati': 'Calories: 240, Protein: 6.2g',
    'dal_makhni': 'Calories: 427, Protein: 24gg',
    'pizza': 'Calories: 570, Protein: 24g',
    'samosa': 'Calories: 261, Protein: 3.5g',
    'kadai_paneer': 'Calories: 302, Protein: 12.3g'
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
    print("Received image from client.")
    try:
        # Decode the image from base64
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform inference using YOLO
        results = model(frame)

        # Process detections and draw bounding boxes and text
        for det in results.xyxy[0]:
            # Access the bounding box coordinates directly from the tensor
            x1, y1, x2, y2, conf, class_index = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4].item(), det[
                5].item()
            class_name = model.names[int(class_index)]

            # Choose color based on class
            box_color = green_box_color if class_name in green_classes else red_box_color

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Prepare and put the text
            text = f'{class_name}: {conf:.2f}'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Add nutritional info if class_name is in the dictionary
            if class_name in nutritional_info:
                info_text = nutritional_info[class_name]
                cv2.putText(frame, info_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Encode the modified frame back to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # Emit the processed image back to the client
        print("Emitting the processed frame.")
        emit('response', {'image': f'data:image/jpeg;base64,{frame_data}'})
    except Exception as e:
        print(f'An error occurred while processing the image: {e}')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000,allow_unsafe_werkzeug=True)
