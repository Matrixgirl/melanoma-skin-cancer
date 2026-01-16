from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pymongo import MongoClient

# Flask setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CNN model
model = load_model(r'C:\Users\Nikita Kodgire\Desktop\mp\aditimp\backend\model\melanoma_cnn.h5')
class_names = ['benign', 'malignant']

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['skin_detection_db']
users_collection = db['users']

# Utility: Check valid file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if users_collection.find_one({'username': username}):
            return render_template('register.html', error='Username already exists')

        users_collection.insert_one({'username': username, 'password': password})
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = users_collection.find_one({'username': username, 'password': password})
        if user:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    confidence = None
    image_url = None
    username = session['user']

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction_result = model.predict(img_array)
            predicted_class = np.argmax(prediction_result)
            confidence = float(np.max(prediction_result))

            prediction = class_names[predicted_class]
            image_url = img_path

    return render_template('dashboard.html', username=username, prediction=prediction,
                           confidence=confidence, image_url=image_url)
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/wikipedia')
def wikipedia():
    return render_template('wikipedia.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video')
def video():
    return render_template('video.html')
@app.route('/performance')
def performance():
    model_metrics = {
        "accuracy": 0.92,
        "loss": 0.28,
        "precision": 0.90,
        "recall": 0.89,
        "f1_score": 0.895,
        "confusion_matrix": [[120, 10], [8, 112]]
    }
    return render_template("performance.html", metrics=model_metrics)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
