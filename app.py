
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import numpy as np
import pydicom
import cv2
from tensorflow import keras
import joblib
from PIL import Image

# Load the saved models
cnn_model = keras.models.load_model('model/cnn_feature_extractor.h5')  
feature_extractor = keras.models.load_model('model/cnn_model.h5')       
scaler = joblib.load('model/scaler.joblib')                             
svm_model = joblib.load('model/svm_model.joblib')   



# Create a label mapping
label_mapping = {
    0: 'Healthy',
    1: 'Herniated_disk',
    2: 'Spinal_stenosis',
    3: 'Spondylosis'
}


def convert_dicom_to_image(dicom_file_path):
    # Load the DICOM file
    dicom_image = pydicom.dcmread(dicom_file_path)
    
    # Convert to a numpy array
    image_array = dicom_image.pixel_array
    
    # Normalize the pixel values to be between 0-255
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
    image_array = image_array.astype(np.uint8)
    
    # Create an image from the numpy array
    image = Image.fromarray(image_array)
    
    # Save the image in a viewable format (e.g., PNG)
    image_file_path = os.path.splitext(dicom_file_path)[0] + '.png'
    image.save(image_file_path)

    return image_file_path

# Function to preprocess a single DICOM image for prediction
def preprocess_image(file_path):
    # Read the DICOM file
    dicom_file = pydicom.dcmread(file_path)
    # Convert to image
    img = dicom_file.pixel_array
    img = cv2.resize(img, (128, 128))  # Resize for consistency
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    img = img.astype('float32') / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to make predictions using all models
def predict_single_input(file_path):
    # Preprocess the image
    input_image = preprocess_image(file_path)

    # Extract features using the CNN
    features = feature_extractor.predict(input_image)

    # Scale the features
    scaled_features = scaler.transform(features)

    # Make predictions using SVM
    svm_prediction = svm_model.predict(scaled_features)

    # Make predictions using CNN
    cnn_prediction = cnn_model.predict(input_image)

    # Get the predicted class
    svm_class = svm_prediction[0]
    cnn_class = np.argmax(cnn_prediction, axis=1)[0]  # Get class index for CNN

    # Map the predicted classes to their corresponding labels
    svm_label = label_mapping[svm_class]
    cnn_label = label_mapping[cnn_class]

    return svm_label, cnn_label

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('auth.html')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return render_template('auth.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)
        
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    svm_label = None  # Initialize the predicted_class variable
    image_path=None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        myfile = request.files['file']

        if myfile.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        fn = myfile.filename

        # Validate DICOM file
        if not fn.lower().endswith('.dcm'):
            flash('Please upload a valid DICOM file.', 'danger')
            return redirect(request.url)

        mypath = os.path.join(r'static/saved_images', fn)
        myfile.save(mypath)
        image_path = convert_dicom_to_image(mypath)
        svm_label, cnn_label = predict_single_input(mypath)

    # Pass the predicted_class and image file name to the template
    return render_template('prediction.html', predicted_class=svm_label, image_path=image_path)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000,debug=True)
