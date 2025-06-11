from flask import Flask, render_template, request, redirect, url_for, flash, session 
from flask_babel import Babel, _
from datetime import datetime
import pandas as pd
import pickle
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
babel = Babel(app)
app.secret_key = "supersecretkey"

# SQLite Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Initialize Database
with app.app_context():
    db.create_all()

# Load datasets and models
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']
model = pickle.load(open('code and dataset/model.pkl', 'rb'))
label_encoders = pickle.load(open('code and dataset/label_encoders.pkl', 'rb'))

destinations_df = pd.read_csv("code and dataset/Expanded_Destinations.csv")
userhistory_df = pd.read_csv("code and dataset/Final_Updated_Expanded_UserHistory.csv")
df = pd.read_csv("code and dataset/final_df.csv")

# Create user-item matrix
user_item_matrix = userhistory_df.pivot(index='UserID', columns='DestinationID', values='ExperienceRating')
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity
user_similarity = cosine_similarity(user_item_matrix)

# Collaborative Filtering Recommendation
def collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df):
    similar_users = user_similarity[user_id - 1]
    similar_users_idx = np.argsort(similar_users)[::-1][1:6]
    similar_user_ratings = user_item_matrix.iloc[similar_users_idx].mean(axis=0)
    recommended_destinations_ids = similar_user_ratings.sort_values(ascending=False).head(10).index
    recommendations = destinations_df[destinations_df['DestinationID'].isin(recommended_destinations_ids)][[
        'DestinationID', 'Name', 'State', 'Type', 'Popularity', 'BestTimeToVisit'
    ]]
    return recommendations

# Predict destination popularity
def recommend_destinations(user_input, model, label_encoders, features, data):
    encoded_input = {}
    for feature in features:
        if feature in label_encoders:
            encoded_input[feature] = label_encoders[feature].transform([user_input[feature]])[0]
        else:
            encoded_input[feature] = user_input[feature]
    input_df = pd.DataFrame([encoded_input])
    predicted_popularity = model.predict(input_df)[0]
    return predicted_popularity

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Destination Selection Page
@app.route('/destination')
def destination():
    return render_template('destination.html')
   
# Register Route
@app.route('/register', methods=['POST'])
def register():
    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]
    hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
    
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash("Email already registered! Please log in.", "danger")
        return redirect(url_for("index"))

    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    flash("Registration successful! Please log in.", "success")
    return redirect(url_for("index"))

# Login Route
@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    if not email or not password:
        flash("Please fill in all fields!", "danger")
        return redirect(url_for("index"))

    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password, password):
        session["user_id"] = user.id
        session["username"] = user.username
        flash("Login successful!", "success")
        return redirect(url_for("recommendation"))
    else:
        flash("Invalid credentials! Please try again.", "danger")

    return redirect(url_for("index"))

# Logout Route
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("Logged out successfully!", "info")
    return redirect(url_for("index"))

# Database Initialization
def init_db():
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS users")  # Delete old table
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                gender TEXT NOT NULL,
                district TEXT NOT NULL,
                state TEXT NOT NULL,
                destination TEXT NOT NULL,
                travel_date TEXT NOT NULL
            )
        """)
        conn.commit()


init_db()  # Ensure database exists on startup
# Recommendation Page

@app.route("/user", methods=["GET"])
def user_details():
    return render_template("user.html")

@app.route("/save_user_details", methods=["POST"])
def save_user_details():
    name = request.form.get("name")
    phone = request.form.get("phone")
    gender = request.form.get("gender")
    district = request.form.get("district")
    state = request.form.get("state")
    destination = request.form.get("destination")
    travel_date = request.form["travel_date"]

    # Save user details in session (or database)
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, phone, gender, district, state, destination, travel_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, phone, gender, district, state, destination, travel_date))
        conn.commit()

    return redirect(url_for("group", destination=destination))


@app.route("/group/<destination>")
def group(destination):
    # Fetch all users for the selected destination and month
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, phone, gender, district, state, destination, travel_date FROM users WHERE destination = ?", (destination,))
        users = cursor.fetchall()  # List of tuples (name, phone, gender, district, state)
        return render_template('group.html', destination=destination, users=users)




@app.route('/recommendation')
def recommendation():
    if "user_id" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("index"))
    return render_template('recommendation.html')

# Recommendation Route
@app.route("/recommend", methods=['POST'])
def recommend():
    if "user_id" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("index"))

    user_id = int(session["user_id"])
    user_input = {
        'Name_x': request.form['name'],
        'Type': request.form['type'],
        'State': request.form['state'],
        'BestTimeToVisit': request.form['best_time'],
        'Preferences': request.form['preferences'],
        'Gender': request.form['gender'],
        'NumberOfAdults': request.form['adults'],
        'NumberOfChildren': request.form['children'],
    }

    recommended_destinations = collaborative_recommend(user_id, user_similarity, user_item_matrix, destinations_df)
    predicted_popularity = recommend_destinations(user_input, model, label_encoders, features, df)

    return render_template('recommendation.html', recommended_destinations=recommended_destinations,
                           predicted_popularity=predicted_popularity)

if __name__ == '__main__':
    app.run(debug=True)
