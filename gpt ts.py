from keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageOps

# Load the model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Create a function to preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert OpenCV BGR format to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    preprocessed_frame = preprocess_image(Image.fromarray(rgb_frame))

    # Expand dimensions to match the model input shape
    data = np.expand_dims(preprocessed_frame, axis=0)

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:]
    confidence_score = prediction[0][index]

    # Overlay prediction on the frame
    label = f"Class: {class_name}, Confidence: {confidence_score:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Classification", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
