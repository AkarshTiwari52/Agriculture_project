from flask import Flask , render_template , url_for, flash ,request , jsonify , redirect
import os
import mysql.connector
import requests
from ml import generate_report , save_prediction_to_db , explain_with_lime , rag_explanation
from moblitnet_dl import generate_image_report
from crop_rec_ml import generate_crop_report , predict_crop_with_lime , final_crop_explaination
# LangChain / LangGraph imports
from langchain.tools import tool
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import numpy as np
from db import get_db
from flask import request, redirect, url_for, flash
from flask_bcrypt import Bcrypt



from tavily import TavilyClient
from typing import Any
import os
TAVILY_API_KEY = "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"

app = Flask(__name__)

from flask_bcrypt import Bcrypt
from flask import session

app.secret_key = "bd5942f08487634d6278cec739f6a16d56c1370f848b3e4aaa4d1a697351c120"
bcrypt = Bcrypt(app)




@app.route("/")
def index():
    return render_template("index.html")



# wheather showcase management okay

@app.route('/wheather')
def wheather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')


    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid=4b07b13a24464a43076fdc75d48f61c9"
    r = requests.get(url).json()

    return jsonify({
        "city": r["name"],
        "temp": r["main"]["temp"],
        "humidity": r["main"]["humidity"],
        "wind": round(r["wind"]["speed"] * 3.6, 1),  # m/s → km/h
        "condition": r["weather"][0]["main"]
    })


# fertilizer area management okay 

@app.route('/fer_rec', methods=['GET', 'POST'])
def fertilizer_recommendation():

    if request.method == 'POST':

        # ---------- Collect input data ----------
        input_data = {
            "Nitrogen": int(request.form['n']),
            "Phosphorous": int(request.form['p']),
            "Potassium": int(request.form['k']),
            "Moisture": float(request.form['moisture']),
            "Temparature": float(request.form['temp']),  # ML spelling preserved
            
            "Crop Type": request.form['crop'],
            "Soil Type": request.form['soil_type'],
            "Humidity": float(request.form['humidity'])
        }

        # ---------- Generate fertilizer report ----------
        result = generate_report(
            crop=input_data["Crop Type"],
            n=input_data["Nitrogen"],
            p=input_data["Phosphorous"],
            k=input_data["Potassium"],
            moisture=input_data["Moisture"],
            temp=input_data["Temparature"],
            humidity=input_data["Humidity"],
            soil_type=input_data["Soil Type"]
        )

        # ---------- ML + XAI ----------
        fert, lime_exp = explain_with_lime(input_data)

        # ---------- RAG Explanation ----------
        rag_text = rag_explanation(fert, lime_exp)

        # ---------- Save to DB ----------
        save_prediction_to_db(
            crop=input_data["Crop Type"],
            n=input_data["Nitrogen"],
            p=input_data["Phosphorous"],
            k=input_data["Potassium"],
            moisture=input_data["Moisture"],
            temp=input_data["Temparature"],
            reccomended_fertlizer=result
        )

        return render_template(
            "fertilizer_reccomendation.html",
            result=result,
            lime=lime_exp,
            rag=rag_text
        )

    return render_template("fertilizer_reccomendation.html")


@app.route('/crop_rec', methods=['GET', 'POST'])
def crop_reccomendation():
    if request.method == 'POST':

        input_data = {
            "N": int(request.form['n']),
            "P": int(request.form['p']),
            "K": int(request.form['k']),
            "temperature": int(request.form['temp']),
            "humidity": float(request.form['humidity']),
            "ph": float(request.form['ph']),
            "rainfall": int(request.form['rainfall'])
        }

        # ML prediction
        result = generate_crop_report(**{
            "n": input_data["N"],
            "p": input_data["P"],
            "k": input_data["K"],
            "temp": input_data["temperature"],
            "humidity": input_data["humidity"],
            "ph": input_data["ph"],
            "rainfall": input_data["rainfall"]
        })

        # XAI
        crop, lime_exp = predict_crop_with_lime(input_data)

        # RAG + LLM explanation
        rag_explanation = final_crop_explaination(crop, lime_exp)

        return render_template(
            "crop_reccomadation.html",
            result=result,
            lime=lime_exp,
            rag=rag_explanation
        )

    return render_template("crop_reccomadation.html")




UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



@app.route('/crop_stress', methods=['GET', 'POST'])
def crop_stress_detection():

    if request.method == 'POST':

        if 'leaf_image' not in request.files:
            return render_template(
                "crop_stress.html",
                error="No image uploaded",
                result=None
            )

        image_file = request.files['leaf_image']

        if image_file.filename == "":
            return render_template(
                "crop_stress.html",
                error="No image selected",
                result=None
            )

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        # 🔥 DL + Grad-CAM + Action Advice
        result = generate_image_report(image_path)

        return render_template(
            "crop_stress.html",
            result=result
        )

    # ✅ SAFE GET REQUEST
    return render_template("crop_stress.html", result=None)



@app.route("/crop_details")
def crop_details():
    return render_template("crop_details.html")



### logiin system for farmers okay

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
# ---------------- DB CONNECTION ----------------
def get_db():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Akarsh2.0@",
        database="agri_auth",
        port=3306
    )



# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT id, full_name, role, password FROM users WHERE email=%s",
            (email,)
        )
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["full_name"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("index"))

        flash("Invalid email or password", "danger")

    return render_template("login.html")


# ================= FARMER SIGNUP =================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form["full_name"]
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        role = "farmer"

        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # check duplicate
        cursor.execute(
            "SELECT id FROM users WHERE username=%s OR email=%s",
            (username, email)
        )
        if cursor.fetchone():
            cursor.close()
            conn.close()
            flash("Username or email already exists", "danger")
            return redirect("/signup")

        cursor.execute("""
            INSERT INTO users (full_name, username, email, password, role)
            VALUES (%s, %s, %s, %s, %s)
        """, (full_name, username, email, password, role))

        conn.commit()
        cursor.close()
        conn.close()

        flash("Account created successfully", "success")
        return redirect("/login")

    return render_template("farmer_signup.html")


# ================= DASHBOARDS =================
#  @app.route("/admin/dashboard")
#def admin_dashboard():
 #   if session.get("role") != "admin":
  #      return redirect(url_for("login"))
#
 #   return render_template("admin_dashboard.html")

@app.route("/farmer_dashboard")
def farmer_dashboard():
    if session.get("role") != "farmer":
        return redirect("/login")
    return render_template("index.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


import re

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([^&?/]+)", url)
    return match.group(1) if match else None


from functools import wraps
from flask import session, redirect, url_for, flash

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Not logged in
        if "user_id" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("login"))

        # Logged in but not admin
        if session.get("role") != "admin":
            flash("Access denied: Admins only", "danger")
            return redirect(url_for("index"))

        return f(*args, **kwargs)
    return decorated_function

def get_db_for_videos():
        return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Akarsh2.0@",
        database="agriculture_videos",
        port=3306
    )
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_for_videos()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM expert_learning_features ORDER BY created_at DESC")
    videos = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("admin_dashboard.html", videos=videos)

@app.route('/admin/upload-video', methods=['POST'])
@admin_required
def upload_video():
    data = request.form
    video_url = data['video_url']
    video_id = extract_video_id(video_url)

    if not video_id:
        flash("❌ Invalid YouTube URL")
        return redirect('/admin/dashboard')

    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO expert_learning_features
        (title, crop, topic, language, expert, video_url, video_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        data['title'],
        data['crop'],
        data['topic'],
        data['language'],
        data['expert'],
        video_url,
        video_id
    ))
    conn.commit()
    cursor.close()
    conn.close()

    flash("✅ Video uploaded successfully")
    return redirect('/admin/dashboard')

@app.route('/admin/delete-video/<int:id>', methods=['POST'])
@admin_required
def delete_video(id):
    conn = mysql.connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expert_learning_features WHERE id=%s", (id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash("🗑 Video deleted successfully")
    return redirect('/admin/dashboard')



@app.route("/learning")
def learning():

    query = request.args.get("q")

    conn = get_db_for_videos()
    cursor = conn.cursor(dictionary=True)

    if query:
        cursor.execute("""
            SELECT * FROM expert_learning_features
            WHERE title LIKE %s
               OR crop LIKE %s
               OR topic LIKE %s
               OR expert LIKE %s
            ORDER BY created_at DESC
        """, tuple(["%" + query + "%"] * 4))
    else:
        cursor.execute("""
            SELECT * FROM expert_learning_features
            ORDER BY created_at DESC
        """)

    videos = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template("learning.html", videos=videos, query=query)





# chatbot system thik hai 
# chatbot config okay 
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")
# Tavily Tool
# -------------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> dict[str, Any]:
    """Search agriculture-related information"""
    return tavily_client.search(query)

# -------------------------------
# System Prompt
# -------------------------------
Base_prompt = """
You are an expert agriculture assistant.
Give practical, region-aware, farmer-friendly advice.
Always ask clarifying questions if data is missing.
Avoid medical or chemical overdose advice.
"""

from flask import request, jsonify
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import uuid

# -------------------------------
# Memory
# -------------------------------
memory = InMemorySaver()

# -------------------------------
# LLM (Groq)
# -------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4
)

# -------------------------------
# Agent (CREATE ONCE)
# -------------------------------
agent = create_agent(
    model=llm,
    system_prompt=Base_prompt,
    tools=[web_search],   # optional
    checkpointer=memory
)

# -------------------------------
# Chat Route
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    language = data.get("language", "English")

    if not user_message:
        return jsonify({"reply": "Please ask a farming question 🌱"})

    thread_id = data.get("thread_id", str(uuid.uuid4()))
    config = {"configurable": {"thread_id": thread_id}}

    try:
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Reply in {language}. {user_message}"
                    )
                ]
            },
            config=config
        )

        bot_reply = response["messages"][-1].content
        return jsonify({"reply": bot_reply, "thread_id": thread_id})

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"reply": "⚠️ Server error. Please try again."})
if __name__ == "__main__":
    app.run(debug=True)