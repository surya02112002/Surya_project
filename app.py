import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import supervision as sv
import os
import uuid  # For generating unique filenames
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import pandas as pd
from joblib import load

# Load skincare products dataset
df = pd.read_csv(r"dataset/updated_skincare_products.csv")

app = Flask(__name__)
app.secret_key = '4545'
DATABASE = 'app.db'

def create_tables():
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS survey_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        age TEXT NOT NULL,
                        gender TEXT NOT NULL,
                        concerns TEXT NOT NULL,
                        acne_frequency TEXT NOT NULL,
                        comedones_count TEXT NOT NULL,
                        first_concern TEXT NOT NULL,
                        cosmetic_usage TEXT NOT NULL,
                        skin_reaction TEXT NOT NULL,
                        skin_type TEXT NOT NULL,
                        medications TEXT NOT NULL,
                        skincare_routine TEXT NOT NULL,
                        stress_level TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users(id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS appointment( 
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT,
                        date TEXT, 
                        skin TEXT,
                        phone TEXT,
                        age TEXT,
                        address TEXT, 
                        status BOOLEAN,
                        username TEXT)''')
        connection.commit()

def insert_user(username, password):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        connection.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()

def insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("""INSERT INTO survey_responses 
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetic_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
            (user_id, name, age, gender, concerns, acne_frequency, comedones_count, first_concern, cosmetics_usage, skin_reaction, skin_type, medications, skincare_routine, stress_level))
        connection.commit()

def insert_appointment_data(name, email, date, skin, phone, age, address, status, username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''INSERT INTO appointment (name, email, date, skin, phone, age, address, status, username)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)''', (name, email, date, skin, phone, age, address, status, username))
        conn.commit()

def findappointment(user):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM appointment WHERE username = ?", (user,))
        appointments = c.fetchall()
    return appointments

def findallappointment():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM appointment")
        appointments = c.fetchall()
    return appointments

def get_survey_response(user_id):
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM survey_responses WHERE user_id = ?", (user_id,))
        return cursor.fetchone()  

def update_appointment_status(appointment_id):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("UPDATE appointment SET status = ? WHERE id = ?", (True, appointment_id))
        conn.commit()

def init_app():
    create_tables()

# Initialize the skin detection model using Roboflow
rf_skin = Roboflow(api_key="8RSJzoEweFB7NxxNK6fg")
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model

# Initialize the oilyness detection model using InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

# Store unique classes detected
unique_classes = set()

# Mapping for oily skin class names
class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"  
}

def recommend_products_based_on_classes(classes):
    recommendations = []
    df_columns_lower = [column.lower() for column in df.columns]
    for skin_condition in classes:
        skin_condition_lower = skin_condition.lower()
        if skin_condition_lower in df_columns_lower:
            original_column = df.columns[df_columns_lower.index(skin_condition_lower)]
            filtered_products = df[df[original_column] == 1][['Brand', 'Name', 'Price', 'Ingredients']]
            filtered_products['Ingredients'] = filtered_products['Ingredients'].apply(lambda x: ', '.join(x.split(', ')[:5]))
            products_list = filtered_products.head(5).to_dict(orient='records')
            recommendations.append((skin_condition, products_list))
        else:
            print(f"Warning: No column found for skin condition '{skin_condition}'")
    return recommendations

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        image_filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join('static', image_filename)
        image_file.save(image_path)

        # Skin detection using Roboflow
        skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
        skin_labels = [item["class"] for item in skin_result["predictions"]]
        for label in skin_labels:
            unique_classes.add(label)

        # Oilyness detection using InferenceHTTPClient
        custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
        with CLIENT.use_configuration(custom_configuration):
            oilyness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")
        
        if not oilyness_result['predictions']:
            unique_classes.add("dryness")
        else:
            oilyness_classes = [class_mapping.get(prediction['class'], prediction['class'])
                                for prediction in oilyness_result['predictions']
                                if prediction['confidence'] >= 0.3]
            for label in oilyness_classes:
                unique_classes.add(label)

        image = cv2.imread(image_path)
        detections = sv.Detections.from_inference(skin_result)
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        recommended_products = recommend_products_based_on_classes(list(unique_classes))
        prediction_data = {
            'classes': list(unique_classes),
            'recommendations': recommended_products
        }
        annotated_image_path = os.path.join('static', 'annotations_0.jpg')
        cv2.imwrite(annotated_image_path, annotated_image)

        return render_template('face_analysis.html', data=prediction_data)
    else:
        return render_template('face_analysis.html', data={})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form.get('name', '')
        age = request.form.get('age', '')
        hashed_password = generate_password_hash(password)

        if get_user(username):
            return "Username already exists. Please choose a different one."
        insert_user(username, hashed_password)
        session['username'] = username
        session['name'] = name
        session['age'] = age
        return redirect('/')
    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            user_id = user[0]
            # Redirect doctor users to the doctor's view
            if username == 'doctor1':
                return redirect(url_for('allappoint'))
            survey_response = get_survey_response(user_id)
            if survey_response:
                return redirect('/profile')
            else:
                return redirect('/survey')
        return "Invalid username or password", 400 
    return render_template('login.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if 'username' not in session:
        return redirect('/')
    if request.method == 'POST':
        user_id = get_user(session['username'])[0]
        name = session.get('name', '')
        age = session.get('age', '')
        gender = request.form['gender']
        concerns = ",".join(request.form.getlist('concerns'))
        acne_frequency = request.form['acne_frequency']
        comedones_count = request.form['comedones_count']
        first_concern = request.form['first_concern']
        cosmetics_usage = request.form['cosmetics_usage']
        skin_reaction = request.form['skin_reaction']
        skin_type = request.form['skin_type_details']
        medications = request.form['medications']
        skincare_routine = request.form['skincare_routine']
        stress_level = request.form['stress_level']

        insert_survey_response(user_id, name, age, gender, concerns, acne_frequency, comedones_count,
                               first_concern, cosmetics_usage, skin_reaction, skin_type,
                               medications, skincare_routine, stress_level)
        return redirect(url_for('profile'))
    return render_template('survey.html', name=session.get('name', ''), age=session.get('age', ''))

@app.route('/profile')
def profile():
    if 'username' in session:
        user_id = get_user(session['username'])[0]
        survey_response = get_survey_response(user_id)
        if survey_response:
            return render_template('profile.html', 
                                name=survey_response[2],
                                age=survey_response[3],
                                gender=survey_response[4],
                                concerns=survey_response[5],
                                acne_frequency=survey_response[6],
                                comedones_count=survey_response[7],
                                first_concern=survey_response[8],
                                cosmetics_usage=survey_response[9],
                                skin_reaction=survey_response[10],
                                skin_type_details=survey_response[11],
                                medications=survey_response[12],
                                skincare_routine=survey_response[13],
                                stress_level=survey_response[14])
    return redirect('/')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/bookappointment')
def bookappointment():
    return render_template('bookappointment.html')

@app.route("/appointment", methods=["POST"])
def appointment():
    name = request.form.get("name")
    email = request.form.get("email")
    date = request.form.get("date")
    skin = request.form.get("skin")
    phone = request.form.get("phone")
    age = request.form.get("age")
    address = request.form.get("address")
    username = session['username']
    status = False
    insert_appointment_data(name, email, date, skin, phone, age, address, status, username)
    return redirect(url_for('bookappointment'))

@app.route("/allappointments")
def allappoint():
    all_appointments = findallappointment()
    # Pass the Python list directly so that we can use the tojson filter in the template.
    return render_template('doctor.html', appointments=all_appointments)

@app.route("/userappointment")
def userappoint():
    user = session['username']
    all_appointments = findappointment(user)
    return render_template('userappointment.html', all_appointments=all_appointments)

@app.route("/update_status", methods=["POST"])
def update_status():
    if request.method == "POST":
        appointment_id = request.form.get("appointment_id")
        update_appointment_status(appointment_id)
        return "updated"

@app.route("/delete_user_request", methods=["POST"])
def delete_user_request():
    if request.method == "POST":
        appointment_id = request.form.get("id")
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM appointment WHERE id = ?", (appointment_id,))
            conn.commit()
        return "deleted successfully"

@app.route("/doctor")
def doctor():
    return render_template('doctor.html')

if __name__ == '__main__':
    init_app()
    app.run(debug=True)
