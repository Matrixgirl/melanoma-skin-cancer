 Project Overview

The Skin Cancer Detection System is a web-based machine learning application designed to assist in the early detection of melanoma skin cancer using image classification techniques. The system allows users to upload skin lesion images and receive predictions based on a trained deep learning model.

This project demonstrates the practical integration of Machine Learning, Python backend development, and web technologies, making it suitable for academic, internship, and entry-level industry evaluation.

 Objectives

Detect melanoma skin cancer using image-based ML models

Provide a simple and interactive web interface for users

Demonstrate real-world application of AI in healthcare

Follow clean project structure suitable for open-source contribution

 Features

Upload skin lesion images for prediction

ML-based classification of melanoma vs non-melanoma

Flask-based backend for handling requests

HTML/CSS frontend for user interaction

Model training script included

Clean GitHub-ready structure (no large datasets committed)

 Tech Stack
Backend

Python

Flask

TensorFlow / Keras

NumPy

OpenCV

Frontend

HTML

CSS

JavaScript

Tools & Concepts

Machine Learning

Image Classification

Git & GitHub

Model Training & Inference

 Project Structure
melanoma-skin-cancer/
│
├── app.py                  # Main Flask application
├── train_model.py          # ML model training script
├── backend/
│   └── requirements.txt    # Python dependencies
│
├── templates/              # HTML templates
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── ...
│
├── static/
│   └── uploads/            # Ignored (runtime uploads only)
│
├── dataset/                # Ignored (local training data)
├── .gitignore
└── README.md

 How to Run the Project
1️ Clone the Repository
git clone https://github.com/Matrixgirl/melanoma-skin-cancer.git
cd melanoma-skin-cancer

2️ Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate

3️ Install Dependencies
pip install -r backend/requirements.txt

4️ Run the Application
python app.py

5️ Open in Browser
http://127.0.0.1:5000/

 Model Training (Optional)

If you want to retrain the model using your own dataset:

python train_model.py


Note: Dataset is intentionally excluded from GitHub to keep the repository lightweight.

 Use Cases

Academic mini / major project

Internship portfolio project

Beginner-friendly ML + Web app

Healthcare AI demonstration

 Future Enhancements

Improve model accuracy with larger datasets

Add multi-class skin disease detection

Deploy on cloud (AWS / Render / Railway)

Add REST API support

Integrate React frontend

 Open Source & Collaboration

This project structure is designed to be open-source friendly and suitable for collaborative development.

 Author

Aditi Chintawar
GitHub: https://github.com/Matrixgirl