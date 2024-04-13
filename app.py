from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the YOLOv8 model with your custom weights
weights_path = 'runs/detect/yolov8m_v8_50bigfood150/weights/best.pt'
model = YOLO(weights_path)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Initialize video stream
# camera = cv2.VideoCapture(0)  # Replace 0 with the correct video source if needed

nutritional_info = {
    'burger': 'Calories: 540, Protein: 34g',
    'chapati': 'Calories: 240, Protein: 6.2g',
    'dal_makhni': 'Calories: 427, Protein: 24gg',
    'pizza': 'Calories: 570, Protein: 24g',
    'samosa': 'Calories: 261, Protein: 3.5g',
    'kadai_paneer': 'Calories: 302, Protein: 12.3g'
    # ... other food items
}

green_box_color = (0, 255, 0)
red_box_color = (0, 0, 255)
text_color = (255, 255, 255)

green_classes = ['dal_makhni', 'kadai_paneer', 'chapati']
red_classes = ['pizza', 'samosa', 'burger']

@app.route('/process_frame', methods=['POST'])
def process_frame():
    image_file = request.files['frame'].read()  # Get the image file
    nparr = np.frombuffer(image_file, np.uint8)  # Convert string to numpy array
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode numpy array to opencv image

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
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({'error': 'Could not encode image.'}), 500

    # Convert to byte array and return as a response
    frame_bytes = buffer.tobytes()
    return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
