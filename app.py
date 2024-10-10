# app.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from flask import Flask, request, render_template
import matplotlib.pyplot as plt

# Initialize the Flask application
app = Flask(__name__)

# Step 1: Load Images from Folder and Preprocess Them
def load_images_from_folder(folder):
    images = []  # List to store the images
    labels = []  # List to store the labels (real or fake)

    for label in os.listdir(folder):
        subfolder = os.path.join(folder, label)
        
        for filename in os.listdir(subfolder):
            try:
                img = Image.open(os.path.join(subfolder, filename)).convert('L')
                img = img.resize((128, 128))
                img = np.array(img) / 255.0
                
                images.append(img)
                labels.append(1 if label == 'real' else 0)  # 1 for real, 0 for fake
            except Exception as e:
                print(f"Error loading image {filename}: {e}")

    images = np.array(images).reshape(-1, 128, 128, 1)
    labels = np.array(labels)
    return images, labels

# Function to build and train the model
def build_and_train_model():
    dataset_folder = 'C:/Users/HARSH RAI/Downloads/Project/train'
    images, labels = load_images_from_folder(dataset_folder)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

    # Save the model
    model.save('currency_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(-1, 128, 128, 1)
    return img

# Route to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file!'

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    model = load_model('currency_detection_model.h5')
    
    prediction = model.predict(img)
    result = 'real' if prediction[0][0] >= 0.5 else 'fake'

    return render_template('result.html', result=result)

# Run the app
if __name__ == '__main__':
    # Check if the model already exists
    if not os.path.exists('currency_detection_model.h5'):
        print("Training the model...")
        build_and_train_model()
    else:
        print("Model already exists. Starting the web app.")
    
    app.run(debug=True)
