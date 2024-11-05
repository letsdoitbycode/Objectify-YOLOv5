from flask import Flask, render_template, Response
import cv2
import torch

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model with verification
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print("Error loading YOLOv5 model:", e)

# Capture from webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

def generate_frames():
    while True:
        # Read the frame
        success, frame = video_capture.read()
        if not success:
            break

        # Perform object detection
        try:
            results = model(frame)  # Run the model on the frame
            print("Detection successful.")

            # Draw bounding boxes and labels
            for result in results.xyxy[0]:  # Iterate through detections
                x1, y1, x2, y2, conf, cls = result
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print("Error during detection:", e)

        # Encode the frame in JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
