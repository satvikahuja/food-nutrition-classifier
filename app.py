from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO

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
camera = cv2.VideoCapture(0)  # Replace 0 with the correct video source if needed
# Attempt to set the resolution to 1920x1080 for high-resolution capture

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

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Process detections
        for det in results[0].boxes:
            # Access the bounding box coordinates directly from the tensor
            if det.xyxy.numel() >= 4:  # Ensure there are at least 4 elements
                coords = det.xyxy[0].tolist()  # Assuming the first element has the bbox coords
                x1, y1, x2, y2 = [int(coord) for coord in coords[:4]]  # Convert to integer if they are not already
            else:
                print("Bounding box coordinates not found")
                continue  # Skip the rest of the loop if coordinates are not found

            # Get the confidence and class index
            conf = det.conf.item()
            class_index = det.cls.item()
            class_name = model.names[int(class_index)]

            # Choose color based on class name
            if class_name in green_classes:
                box_color = green_box_color
            elif class_name in red_classes:
                box_color = red_box_color
            else:
                continue

                # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 6)

                # Prepare and put the text
            text = f'{class_name}: {conf:.2f}'
            cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

                # Add nutritional info if class_name is in the dictionary
            if class_name in nutritional_info:
                info_text = nutritional_info[class_name]
                cv2.putText(frame, info_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, .75, text_color, 2)


        # Encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        # Yield the output frame in byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#update
if __name__ == '__main__':
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
