import streamlit as st
import cv2
import os
from deepface import DeepFace
import numpy as np
from PIL import Image
import io

# Folder to save unique faces
FACE_FOLDER = "faces"
os.makedirs(FACE_FOLDER, exist_ok=True)

def get_face_embedding(face_image):
    """Extract the embedding for the face image using DeepFace."""
    embedding = DeepFace.represent(face_image, model_name="ArcFace", enforce_detection=False)
    return embedding[0]["embedding"]

def are_faces_same(embedding1, embedding2, threshold=0.6):
    """Compare two face embeddings using cosine similarity."""
    from scipy.spatial.distance import cosine
    similarity = cosine(embedding1, embedding2)
    return similarity < threshold  # A lower value indicates higher similarity

def save_unique_face(face_image, saved_embeddings):
    """Save the face image if it's not already saved."""
    face_embedding = get_face_embedding(face_image)
    
    for saved_embedding in saved_embeddings:
        if are_faces_same(face_embedding, saved_embedding):
            print("Duplicate face detected, not saving.")
            return

    # Save the unique face
    filename = os.path.join(FACE_FOLDER, f"face_{len(saved_embeddings) + 1}.jpg")
    cv2.imwrite(filename, face_image)
    saved_embeddings.append(face_embedding)  # Add the embedding to the saved list
    print(f"Saved new face: {filename}")

def detect_and_save_faces(frame, saved_embeddings):
    """Detect faces, extract them, and save unique ones."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]

        # Save the unique face if it's not a duplicate
        save_unique_face(face_image, saved_embeddings)

    return frame

def main():
    st.title("Face Recognition and Detection App")

    # Camera input widget for Streamlit
    camera_input = st.camera_input("Take a picture of a face")

    # List to store saved embeddings
    saved_embeddings = []

    if camera_input:
        # Convert the image from camera input (in byte format) to PIL Image
        image = Image.open(io.BytesIO(camera_input.getvalue()))
        image = np.array(image)  # Convert to numpy array for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR

        # Detect faces and save unique ones
        frame_with_faces = detect_and_save_faces(image, saved_embeddings)

        # Display the resulting frame with the detected faces
        st.image(frame_with_faces, channels="BGR", caption="Detected Faces")

if __name__ == "__main__":
    main()