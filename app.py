from flask import Flask, render_template, Response
import torch
from PIL import Image
from torchvision.transforms import functional as F
from utils.general import non_max_suppression
from models.experimental import attempt_load
import cv2

app = Flask(__name__)
# Load YOLOv5 model
weights_path = "model_yolov5m.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights_path)
model = model.to(device).eval()

def preprocess_image(image):
    # Preprocess the image for YOLOv5
    image = F.resize(image, (640, 640))
    image = F.pad(image, (0, 0, 0, 0), fill=0)
    image = F.to_tensor(image)
    image = image.unsqueeze(0)
    return image

def run_yolov5_inference(image):
    # Run YOLOv5 inference on the image
    image = image.to(device)
    output = model(image)[0]
    output = non_max_suppression(output, conf_thres=0.6, iou_thres=0.5)
    return output

def generate_frames():
    # Open the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = preprocess_image(image)

        # Run YOLOv5 inference on the frame
        output = run_yolov5_inference(image)

        # Visualize the detections on the frame
        for detection in output[0]:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers

            # Calculate the center coordinates of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate the width and height of the bounding box
            width = x2 - x1
            height = y2 - y1

            # Calculate the adjusted bounding box coordinates
            new_width = int(width * 1.2)  # Adjusted width
            new_height = int(height * 1.2)  # Adjusted height

            x1 = max(center_x - new_width // 2, 0)
            y1 = max(center_y - new_height // 2, 0)
            x2 = min(center_x + new_width // 2, frame.shape[1])
            y2 = min(center_y + new_height // 2, frame.shape[0])

            # Calculate the object's coordinates within the adjusted bounding box
            obj_x1 = max(center_x - width // 2, 0)
            obj_y1 = max(center_y - height // 2, 0)
            obj_x2 = min(center_x + width // 2, frame.shape[1])
            obj_y2 = min(center_y + height // 2, frame.shape[0])

            # Draw the adjusted bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the object inside the adjusted bounding box
           # cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{cls}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to Flask for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera and cleanup
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
