import cv2
import os
from inference_sdk import InferenceHTTPClient

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not API_KEY:
    print("Set ROBOFLOW_API_KEY env var first")
    exit(1)

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Grab a single frame from camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read camera frame")
    exit(1)

cv2.imwrite("test_frame.jpg", frame)
print(f"Captured frame: {frame.shape}")
print("Sending to Roboflow...")

# Test 1: numpy array
try:
    result = CLIENT.infer(frame, model_id="asl-alphabet-recognition/7")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
except Exception as e:
    print(f"numpy array failed: {e}")

# Test 2: file path
try:
    result2 = CLIENT.infer("test_frame.jpg", model_id="asl-alphabet-recognition/7")
    print(f"File path result: {result2}")
except Exception as e:
    print(f"file path failed: {e}")
