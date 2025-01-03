# facedetect
Here is a README.md file that you can use for your project:

Face Recognition and Detection App

This is a face recognition and detection app built using OpenCV, DeepFace, and Streamlit. The app detects faces from uploaded images or from your camera feed, extracts the face embeddings, and saves only unique faces to a folder.

Features
	•	Face Detection: Detects faces from images or camera feed using OpenCV’s pre-trained Haar cascade.
	•	Face Embedding: Uses the DeepFace library (with the ArcFace model) to generate face embeddings, which are used for face similarity comparison.
	•	Save Unique Faces: Only saves a face if it’s not already stored, based on similarity comparison.
	•	Streamlit Interface: The app is built as a web application using Streamlit for easy interaction.

Requirements

Make sure you have the following libraries installed in your environment:
	•	streamlit
	•	opencv-python
	•	deepface
	•	numpy
	•	PIL
	•	scipy

You can install them using pip:

pip install streamlit opencv-python deepface numpy Pillow scipy

Setup
	1.	Clone the Repository (if using Git):

git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app


	2.	Run the App:
After setting up your environment and installing dependencies, navigate to the project folder and run:

streamlit run face_recognition_app.py

This will start the Streamlit web app in your browser.

	3.	Folder for Storing Faces:
	•	The app creates a folder called faces in the working directory to store the unique faces that are detected. Each unique face will be saved as an image in this folder.

How It Works
	1.	Face Detection:
	•	When you upload an image or use the camera, the app first detects faces in the frame using OpenCV’s Haar Cascade classifier.
	2.	Face Embedding Generation:
	•	After detecting a face, the app uses DeepFace with the ArcFace model to generate a unique embedding (vector representation) of the face.
	3.	Face Comparison:
	•	The app checks whether the face embedding already exists in the saved_embeddings list by comparing it with existing embeddings using cosine similarity.
	4.	Saving Unique Faces:
	•	If the detected face is not a duplicate (based on the similarity threshold), it will be saved in the faces folder, and its embedding will be added to the list of saved embeddings.

How to Use
	•	Upload or Capture Image: Use the camera input feature in Streamlit to either upload an image or capture a picture of your face.
	•	Detection and Saving: The app will automatically detect faces in the uploaded image and save unique faces to the faces folder. Duplicate faces will be skipped.

Customization
	•	Threshold: You can adjust the similarity threshold used to compare faces. The default is 0.6, meaning two faces are considered the same if their cosine similarity is below this value.
	•	Face Embedding Model: The app uses the ArcFace model from DeepFace. You can change the model to others supported by DeepFace (e.g., VGG-Face, Facenet, OpenFace, etc.).

Example

After running the app, you will see a camera feed or file upload button. When you take or upload an image, the app will detect the face, compare it with previously saved faces, and store unique ones.

Output

The faces that are detected will be displayed in the Streamlit interface, and new unique faces will be saved in the faces folder. You can navigate to the folder to check the saved face images.

License

This project is licensed under the MIT License.

Acknowledgements
	•	OpenCV: For face detection using Haar cascades.
	•	DeepFace: For generating face embeddings and comparing them.
	•	Streamlit: For creating the interactive web interface.
	•	ArcFace: The model used for face recognition.

You can modify the URLs or other details as necessary, and make sure to replace "yourusername" in the clone command with your actual GitHub username if you’re sharing this repository.