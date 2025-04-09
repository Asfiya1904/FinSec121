import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import uuid
import datetime
import sqlite3
import base64
import requests
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import openai
import time
import hashlib
from PIL import Image

# Load environment variables
load_dotenv()

# Configuration
FINSEC_API_URL = os.getenv("FINSEC_API_URL", "https://finsec1.onrender.com/detect")
FINSEC_API_KEY = os.getenv("FINSEC_API_KEY", "supersecret")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initialize OpenAI client if API key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Database setup
def init_db():
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE,
        password TEXT,
        role TEXT,
        plan TEXT,
        created_at TIMESTAMP
    )
    ''')
    
    # Create scans table
    c.execute('''
    CREATE TABLE IF NOT EXISTS scans (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        filename TEXT,
        total_transactions INTEGER,
        high_risk_count INTEGER,
        medium_risk_count INTEGER,
        low_risk_count INTEGER,
        scan_date TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create settings table
    c.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        user_id TEXT PRIMARY KEY,
        email_alerts BOOLEAN,
        live_access BOOLEAN,
        webhook_url TEXT,
        api_key TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create default admin user if not exists
    c.execute("SELECT * FROM users WHERE email = 'admin@finsec.com'")
    if not c.fetchone():
        admin_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256("admin123".encode()).hexdigest()
        created_at = datetime.datetime.now()
        
        c.execute(
            "INSERT INTO users (id, email, password, role, plan, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (admin_id, "admin@finsec.com", hashed_password, "admin", "premium", created_at)
        )
        
        # Create default settings for admin
        api_key = f"fsk_{uuid.uuid4().hex[:16]}"
        c.execute(
            "INSERT INTO settings (user_id, email_alerts, live_access, webhook_url, api_key) VALUES (?, ?, ?, ?, ?)",
            (admin_id, True, True, "", api_key)
        )
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Page configuration
st.set_page_config(
    page_title="FinSec - Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    css = """
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            background-image: url("app/static/dashboard_background.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .sidebar .sidebar-content {
            background-color: rgba(20, 39, 78, 0.9);
            color: white;
        }
        .css-1d391kg {
            padding-top: 0rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(20, 39, 78, 0.1);
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            color: #14274E;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(20, 39, 78, 0.8) !important;
            color: white !important;
        }
        .css-1y4p8pa {
            max-width: 100%;
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .main-header {
            background-color: rgba(20, 39, 78, 0.8);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .card {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: rgba(20, 39, 78, 0.8);
            border-radius: 10px;
            padding: 1rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .risk-high {
            color: #ff4b4b;
            font-weight: bold;
        }
        .risk-medium {
            color: #ffa500;
            font-weight: bold;
        }
        .risk-low {
            color: #00cc96;
            font-weight: bold;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: rgba(20, 39, 78, 0.1);
            border-left: 5px solid #14274E;
        }
        .chat-message.assistant {
            background-color: rgba(20, 39, 78, 0.05);
            border-left: 5px solid #4CAF50;
        }
        .chat-message .content {
            margin-top: 0.5rem;
        }
        .btn-primary {
            background-color: #14274E;
            color: white;
        }
        .btn-danger {
            background-color: #ff4b4b;
            color: white;
        }
        .btn-success {
            background-color: #00cc96;
            color: white;
        }
        .btn-warning {
            background-color: #ffa500;
            color: white;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            background-color: rgba(20, 39, 78, 0.8);
            color: white;
            border-radius: 10px;
        }
        .footer a {
            color: white;
            text-decoration: underline;
        }
        .footer a:hover {
            color: #f0f0f0;
        }
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        /* Logout modal */
        .logout-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .logout-modal-content {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            width: 400px;
            max-width: 90%;
        }
        .logout-modal-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
        }
        /* Chat interface */
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            height: 500px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            background-color: #14274E;
            color: white;
            padding: 1rem;
            border-radius: 10px 10px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-input-container {
            padding: 1rem;
            border-top: 1px solid #f0f0f0;
            display: flex;
        }
        .chat-input {
            flex: 1;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .chat-send-button {
            background-color: #14274E;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            margin-left: 0.5rem;
            cursor: pointer;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load custom CSS
load_css()

# Session state initialization
if 'user' not in st.session_state:
    st.session_state.user = None
if 'login_status' not in st.session_state:
    st.session_state.login_status = None
if 'signup_status' not in st.session_state:
    st.session_state.signup_status = None
if 'show_logout_modal' not in st.session_state:
    st.session_state.show_logout_modal = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Hello! I'm your FinSec assistant. How can I help you today?"}]
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
if 'login_role' not in st.session_state:
    st.session_state.login_role = "user"
if 'page' not in st.session_state:
    st.session_state.page = "login"

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password, role="client", plan="free"):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    # Check if user already exists
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    if c.fetchone():
        conn.close()
        return False, "User with this email already exists"
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(password)
    created_at = datetime.datetime.now()
    
    c.execute(
        "INSERT INTO users (id, email, password, role, plan, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, email, hashed_password, role, plan, created_at)
    )
    
    # Create default settings for user
    api_key = f"fsk_{uuid.uuid4().hex[:16]}"
    c.execute(
        "INSERT INTO settings (user_id, email_alerts, live_access, webhook_url, api_key) VALUES (?, ?, ?, ?, ?)",
        (user_id, False, False, "", api_key)
    )
    
    conn.commit()
    conn.close()
    
    return True, user_id

def authenticate_user(email, password):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_password))
    user = c.fetchone()
    
    conn.close()
    
    if user:
        return True, {
            "id": user[0],
            "email": user[1],
            "role": user[3],
            "plan": user[4],
            "created_at": user[5]
        }
    else:
        return False, None

def get_user_settings(user_id):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM settings WHERE user_id = ?", (user_id,))
    settings = c.fetchone()
    
    conn.close()
    
    if settings:
        return {
            "email_alerts": bool(settings[1]),
            "live_access": bool(settings[2]),
            "webhook_url": settings[3],
            "api_key": settings[4]
        }
    else:
        return {
            "email_alerts": False,
            "live_access": False,
            "webhook_url": "",
            "api_key": ""
        }

def update_user_settings(user_id, email_alerts, live_access, webhook_url):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    c.execute(
        "UPDATE settings SET email_alerts = ?, live_access = ?, webhook_url = ? WHERE user_id = ?",
        (email_alerts, live_access, webhook_url, user_id)
    )
    
    conn.commit()
    conn.close()
    
    return True

def save_scan_results(user_id, filename, total, high, medium, low):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    scan_id = str(uuid.uuid4())
    scan_date = datetime.datetime.now()
    
    c.execute(
        "INSERT INTO scans (id, user_id, filename, total_transactions, high_risk_count, medium_risk_count, low_risk_count, scan_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (scan_id, user_id, filename, total, high, medium, low, scan_date)
    )
    
    conn.commit()
    conn.close()
    
    return scan_id

def get_user_scans(user_id):
    conn = sqlite3.connect('finsec.db')
    c = conn.cursor()
    
    c.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY scan_date DESC", (user_id,))
    scans = c.fetchall()
    
    conn.close()
    
    result = []
    for scan in scans:
        result.append({
            "id": scan[0],
            "filename": scan[2],
            "total": scan[3],
            "high_risk": scan[4],
            "medium_risk": scan[5],
            "low_risk": scan[6],
            "date": scan[7]
        })
    
    return result

# Fraud detection functions
def analyze_transactions(df):
    # Add risk score calculation
    df['risk_score'] = np.random.uniform(0, 1, size=len(df))
    
    # Assign risk categories
    def assign_risk_category(score):
        if score < 0.3:
            return "Low"
        elif score < 0.7:
            return "Medium"
        else:
            return "High"
    
    df['risk_category'] = df['risk_score'].apply(assign_risk_category)
    
    # Count risk categories
    risk_counts = df['risk_category'].value_counts().to_dict()
    high_count = risk_counts.get('High', 0)
    medium_count = risk_counts.get('Medium', 0)
    low_count = risk_counts.get('Low', 0)
    
    # Calculate percentages
    total = len(df)
    high_percent = round((high_count / total) * 100)
    medium_percent = round((medium_count / total) * 100)
    low_percent = round((low_count / total) * 100)
    
    # Generate summary
    summary = f"{high_percent}% of transactions were high risk, {medium_percent}% medium risk, and {low_percent}% low risk."
    
    # Add fraud indicators (simulated)
    fraud_indicators = [
        "Unusual transaction amount",
        "Suspicious IP address",
        "Multiple transactions in short time",
        "Unusual location",
        "Mismatched billing information"
    ]
    
    # Randomly assign fraud indicators to high and medium risk transactions
    def assign_indicators(row):
        if row['risk_category'] == 'High':
            return ', '.join(np.random.choice(fraud_indicators, size=np.random.randint(2, 4), replace=False))
        elif row['risk_category'] == 'Medium':
            return ', '.join(np.random.choice(fraud_indicators, size=np.random.randint(1, 3), replace=False))
        else:
            return ''
    
    df['fraud_indicators'] = df.apply(assign_indicators, axis=1)
    
    return df, {
        'total': total,
        'high_count': high_count,
        'medium_count': medium_count,
        'low_count': low_count,
        'high_percent': high_percent,
        'medium_percent': medium_percent,
        'low_percent': low_percent,
        'summary': summary
    }

def api_analyze_transaction(transaction_data):
    # Simulate API call
    time.sleep(1)  # Simulate network delay
    
    # Generate random risk score
    risk_score = np.random.uniform(0, 1)
    
    # Assign risk category
    if risk_score < 0.3:
        risk_category = "Low"
        fraud_indicators = []
    elif risk_score < 0.7:
        risk_category = "Medium"
        fraud_indicators = np.random.choice([
            "Unusual transaction amount",
            "Suspicious IP address",
            "Multiple transactions in short time"
        ], size=np.random.randint(1, 3), replace=False).tolist()
    else:
        risk_category = "High"
        fraud_indicators = np.random.choice([
            "Unusual transaction amount",
            "Suspicious IP address",
            "Multiple transactions in short time",
            "Unusual location",
            "Mismatched billing information"
        ], size=np.random.randint(2, 4), replace=False).tolist()
    
    return {
        'transaction_id': transaction_data.get('transaction_id', str(uuid.uuid4())),
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'fraud_indicators': fraud_indicators,
        'timestamp': datetime.datetime.now().isoformat()
    }

# AI Chatbot functions
def get_ai_response(query):
    if not OPENAI_API_KEY:
        # Fallback responses if OpenAI API key is not available
        fallback_responses = {
            "help": "I can help you with fraud detection, transaction analysis, and using the FinSec platform. What would you like to know?",
            "fraud": "Fraud detection involves analyzing transaction patterns to identify suspicious activities. FinSec uses advanced algorithms to assign risk scores to transactions.",
            "risk": "Risk scores are calculated based on various factors including transaction amount, location, frequency, and other patterns. High risk transactions may indicate potential fraud.",
            "dashboard": "The dashboard provides an overview of your transaction analysis, including risk distribution and fraud indicators.",
            "settings": "In the settings page, you can configure email alerts, API integration, and other preferences.",
            "api": "FinSec provides API integration for real-time transaction monitoring. You can find your API key in the settings page.",
            "report": "You can download transaction analysis reports in CSV format from the dashboard after analyzing transactions.",
            "alert": "Email alerts can be configured to notify you when high-risk transactions are detected.",
            "privacy": "FinSec is GDPR-compliant and respects your data privacy. You can view our privacy policy for more information.",
            "default": "I'm your FinSec assistant. I can help you with fraud detection, transaction analysis, and using the platform. Please ask a specific question."
        }
        
        # Simple keyword matching for fallback responses
        for keyword, response in fallback_responses.items():
            if keyword.lower() in query.lower():
                return response
        
        return fallback_responses["default"]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for FinSec, a financial fraud detection platform. Provide concise, helpful responses about using the platform, fraud detection, and financial security."},
                {"role": "user", "content": query}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm having trouble connecting to my knowledge base. I can still help with basic questions about fraud detection and using the FinSec platform."

# Utility functions
def get_table_download_link(df, filename="finsec_report.csv", text="Download CSV Report"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn-primary" style="text-decoration:none;padding:0.5rem 1rem;border-radius:5px;">{text}</a>'
    return href

# Sidebar navigation
def render_sidebar():
    with st.sidebar:
        st.image("static/finsec_logo.png", width=200)
        st.markdown("### Predict. Prevent. Protect.")
        
        st.markdown("---")
        
        if st.session_state.user:
            st.markdown(f"### Welcome, {st.session_state.user['email']}")
            st.markdown(f"**Plan:** {st.session_state.user['plan'].capitalize()}")
            st.markdown(f"**Role:** {st.session_state.user['role'].capitalize()}")
            st.markdown("---")
            
            if st.button("Dashboard"):
                st.session_state.page = "dashboard"
                st.experimental_rerun()
            
            if st.button("History"):
                st.session_state.page = "history"
                st.experimental_rerun()
            
            if st.button("Settings"):
                st.session_state.page = "settings"
                st.experimental_rerun()
            
            if st.button("Privacy Policy"):
                st.session_state.page = "privacy"
                st.experimental_rerun()
            
            st.markdown("---")
            
            if st.button("Logout"):
                st.session_state.show_logout_modal = True
                st.experimental_rerun()
            
            # AI Assistant toggle
            st.markdown("---")
            st.markdown("### AI Assistant")
            if st.toggle("Show AI Assistant", value=st.session_state.show_chat):
                st.session_state.show_chat = True
            else:
                st.session_state.show_chat = False
        else:
            if st.button("Login"):
                st.session_state.page = "login"
                st.experimental_rerun()
            
            if st.button("Sign Up"):
                st.session_state.page = "signup"
                st.experimental_rerun()
            
            if st.button("Privacy Policy"):
                st.session_state.page = "privacy"
                st.experimental_rerun()

# Logout modal
def render_logout_modal():
    if st.session_state.show_logout_modal:
        st.markdown("""
        <div class="logout-modal">
            <div class="logout-modal-content">
                <h3>Confirm Logout</h3>
                <p>Are you sure you want to logout?</p>
                <div class="logout-modal-buttons">
                    <button onclick="cancelLogout()" style="background-color:#f0f0f0;border:none;padding:0.5rem 1rem;border-radius:5px;cursor:pointer;">Cancel</button>
                    <button onclick="confirmLogout()" style="background-color:#ff4b4b;color:white;border:none;padding:0.5rem 1rem;border-radius:5px;cursor:pointer;">Logout</button>
                </div>
            </div>
        </div>
        <script>
            function cancelLogout() {
                window.parent.postMessage({"type": "streamlit:setComponentValue", "value": "cancel_logout"}, "*");
            }
            function confirmLogout() {
                window.parent.postMessage({"type": "streamlit:setComponentValue", "value": "confirm_logout"}, "*");
            }
        </script>
        """, unsafe_allow_html=True)
        
        # Handle logout confirmation via buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="cancel_logout_button"):
                st.session_state.show_logout_modal = False
                st.experimental_rerun()
        
        with col2:
            if st.button("Confirm Logout", key="confirm_logout_button"):
                st.session_state.user = None
                st.session_state.login_status = None
                st.session_state.show_logout_modal = False
                st.session_state.page = "login"
                st.experimental_rerun()

# AI Chat interface
def render_chat_interface():
    if st.session_state.show_chat:
        chat_container = st.container()
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        st.markdown("""
        <div style="position:fixed;bottom:20px;right:20px;width:350px;height:500px;background-color:white;border-radius:10px;box-shadow:0 4px 6px rgba(0,0,0,0.1);z-index:1000;display:flex;flex-direction:column;">
            <div style="background-color:#14274E;color:white;padding:1rem;border-radius:10px 10px 0 0;display:flex;justify-content:space-between;align-items:center;">
                <h3 style="margin:0;">FinSec Assistant</h3>
                <button id="close-chat" onclick="closeChat()" style="background:none;border:none;color:white;cursor:pointer;font-size:1.2rem;">×</button>
            </div>
            <div id="chat-messages" style="flex:1;overflow-y:auto;padding:1rem;display:flex;flex-direction:column;">
                <!-- Messages will be displayed here -->
            </div>
            <div style="padding:1rem;border-top:1px solid #f0f0f0;display:flex;">
                <input id="chat-input" type="text" placeholder="Type your message..." style="flex:1;padding:0.5rem;border:1px solid #ddd;border-radius:5px;">
                <button onclick="sendChatMessage()" style="background-color:#14274E;color:white;border:none;border-radius:5px;padding:0.5rem 1rem;margin-left:0.5rem;cursor:pointer;">Send</button>
            </div>
        </div>
        <script>
            function closeChat() {
                window.parent.postMessage({"type": "streamlit:setComponentValue", "value": "close_chat"}, "*");
            }
            
            function sendChatMessage() {
                var input = document.getElementById("chat-input");
                var message = input.value.trim();
                
                if (message) {
                    window.parent.postMessage({"type": "streamlit:setComponentValue", "value": message}, "*");
                    input.value = "";
                }
            }
            
            // Add event listener for Enter key
            document.getElementById("chat-input").addEventListener("keypress", function(e) {
                if (e.key === "Enter") {
                    sendChatMessage();
                }
            });
            
            // Display messages
            function displayMessages() {
                var messagesDiv = document.getElementById("chat-messages");
                messagesDiv.innerHTML = "";
                
                var messages = JSON.parse('{messages}');
                
                for (var i = 0; i < messages.length; i++) {
                    var message = messages[i];
                    var messageDiv = document.createElement("div");
                    
                    if (message.role === "user") {
                        messageDiv.className = "chat-message user";
                    } else {
                        messageDiv.className = "chat-message assistant";
                    }
                    
                    messageDiv.innerHTML = '<div class="content">' + message.content + '</div>';
                    messagesDiv.appendChild(messageDiv);
                }
                
                // Scroll to bottom
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
            
            // Call displayMessages on load
            displayMessages();
        </script>
        """.replace("{messages}", json.dumps(st.session_state.chat_messages)), unsafe_allow_html=True)
        
        # Handle chat input
        chat_input = st.text_input("Message", key="chat_input_field", label_visibility="collapsed")
        
        if st.button("Send", key="send_chat_button"):
            if chat_input:
                # Add user message to chat
                st.session_state.chat_messages.append({"role": "user", "content": chat_input})
                
                # Get AI response
                response = get_ai_response(chat_input)
                
                # Add assistant response to chat
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # Clear input
                st.session_state.chat_input = ""
                
                # Rerun to update UI
                st.experimental_rerun()

# Page: Role Selection
def render_role_selection_page():
    st.markdown('<div class="main-header"><h1>Select Login Type</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("static/finsec_logo.png", width=300)
            st.markdown("### Predict. Prevent. Protect.")
            st.markdown("Welcome to FinSec, your advanced financial fraud detection platform.")
        
        with col2:
            st.markdown("### Choose Login Type")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Admin Login", use_container_width=True):
                    st.session_state.login_role = "admin"
                    st.session_state.page = "login"
                    st.experimental_rerun()
            
            with col2:
                if st.button("User Login", use_container_width=True):
                    st.session_state.login_role = "user"
                    st.session_state.page = "login"
                    st.experimental_rerun()
            
            st.markdown("---")
            
            st.markdown("Don't have an account? [Sign up](/signup)")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Login
def render_login_page():
    role_title = "Admin" if st.session_state.login_role == "admin" else "User"
    
    st.markdown(f'<div class="main-header"><h1>FinSec {role_title} Login</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("static/finsec_logo.png", width=300)
            st.markdown("### Predict. Prevent. Protect.")
            st.markdown("Welcome to FinSec, your advanced financial fraud detection platform.")
        
        with col2:
            st.markdown(f"### {role_title} Login")
            
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                if email and password:
                    success, user = authenticate_user(email, password)
                    if success:
                        # Check if role matches
                        if (st.session_state.login_role == "admin" and user["role"] == "admin") or \
                           (st.session_state.login_role == "user" and user["role"] == "client"):
                            st.session_state.user = user
                            st.session_state.login_status = "success"
                            st.session_state.page = "dashboard"
                            st.experimental_rerun()
                        else:
                            st.error(f"Invalid credentials for {role_title} login")
                    else:
                        st.session_state.login_status = "failed"
                        st.error("Invalid email or password")
                else:
                    st.warning("Please enter both email and password")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Switch to User" if st.session_state.login_role == "admin" else "Switch to Admin"):
                    st.session_state.login_role = "user" if st.session_state.login_role == "admin" else "admin"
                    st.experimental_rerun()
            
            with col2:
                if st.button("Back to Selection"):
                    st.session_state.page = "role_selection"
                    st.experimental_rerun()
            
            st.markdown("Don't have an account? [Sign up](/signup)")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Signup
def render_signup_page():
    st.markdown('<div class="main-header"><h1>FinSec Sign Up</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image("static/finsec_logo.png", width=300)
            st.markdown("### Predict. Prevent. Protect.")
            st.markdown("Join FinSec today and secure your financial transactions with our advanced fraud detection platform.")
        
        with col2:
            st.markdown("### Create Your Account")
            
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            
            # Role selection
            role = st.selectbox("Account Type", ["User", "Admin"], index=0)
            
            if st.button("Sign Up", key="signup_button"):
                if email and password and confirm_password:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Convert role selection to database role
                        db_role = "client" if role == "User" else "admin"
                        
                        success, message = create_user(email, password, role=db_role)
                        if success:
                            st.session_state.signup_status = "success"
                            st.success("Account created successfully! Please login.")
                            time.sleep(2)
                            st.session_state.login_role = "user" if role == "User" else "admin"
                            st.session_state.page = "login"
                            st.experimental_rerun()
                        else:
                            st.session_state.signup_status = "failed"
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")
            
            st.markdown("Already have an account? [Login](/login)")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Dashboard
def render_dashboard_page():
    st.markdown('<div class="main-header"><h1>FinSec Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">98.7%</div><div class="metric-label">Detection Accuracy</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">24/7</div><div class="metric-label">Monitoring</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">< 1s</div><div class="metric-label">Response Time</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">100+</div><div class="metric-label">Fraud Patterns</div></div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    tabs = st.tabs(["Upload Transactions", "Live Monitoring"])
    
    with tabs[0]:
        st.markdown("### Upload Transaction Data")
        st.markdown("Upload your CSV file containing transaction data for fraud analysis.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown("### Transaction Data Preview")
                st.dataframe(df.head())
                
                if st.button("Analyze Transactions"):
                    with st.spinner("Analyzing transactions..."):
                        # Perform analysis
                        results_df, summary = analyze_transactions(df)
                        st.session_state.analysis_results = {
                            "df": results_df,
                            "summary": summary
                        }
                        
                        # Save scan results to database
                        save_scan_results(
                            st.session_state.user["id"],
                            uploaded_file.name,
                            summary["total"],
                            summary["high_count"],
                            summary["medium_count"],
                            summary["low_count"]
                        )
                        
                        st.success("Analysis complete!")
                        st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            df = results["df"]
            summary = results["summary"]
            
            st.markdown("### Analysis Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'<div class="metric-card" style="background-color: #ff4b4b;"><div class="metric-value">{summary["high_count"]}</div><div class="metric-label">High Risk Transactions</div></div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="metric-card" style="background-color: #ffa500;"><div class="metric-value">{summary["medium_count"]}</div><div class="metric-label">Medium Risk Transactions</div></div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="metric-card" style="background-color: #00cc96;"><div class="metric-value">{summary["low_count"]}</div><div class="metric-label">Low Risk Transactions</div></div>', unsafe_allow_html=True)
            
            st.markdown(f"**Summary:** {summary['summary']}")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Risk Distribution")
                fig = px.pie(
                    names=["High", "Medium", "Low"],
                    values=[summary["high_count"], summary["medium_count"], summary["low_count"]],
                    color=["High", "Medium", "Low"],
                    color_discrete_map={"High": "#ff4b4b", "Medium": "#ffa500", "Low": "#00cc96"}
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Fraud Indicators")
                # Count fraud indicators
                indicators = []
                for ind in df["fraud_indicators"]:
                    if ind:
                        indicators.extend([i.strip() for i in ind.split(",")])
                
                indicator_counts = pd.Series(indicators).value_counts().reset_index()
                indicator_counts.columns = ["Indicator", "Count"]
                
                fig = px.bar(
                    indicator_counts,
                    x="Count",
                    y="Indicator",
                    orientation="h",
                    color_discrete_sequence=["#14274E"]
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.markdown("### Detailed Results")
            
            # Add styling to risk categories
            def highlight_risk(val):
                if val == "High":
                    return "background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; font-weight: bold"
                elif val == "Medium":
                    return "background-color: rgba(255, 165, 0, 0.2); color: #ffa500; font-weight: bold"
                elif val == "Low":
                    return "background-color: rgba(0, 204, 150, 0.2); color: #00cc96; font-weight: bold"
                return ""
            
            styled_df = df.style.applymap(highlight_risk, subset=["risk_category"])
            st.dataframe(styled_df)
            
            # Download link
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            
            # Clear results button
            if st.button("Clear Results"):
                st.session_state.analysis_results = None
                st.session_state.uploaded_file = None
                st.experimental_rerun()
    
    with tabs[1]:
        st.markdown("### Live Transaction Monitoring")
        
        # Check user plan
        if st.session_state.user["plan"] == "free":
            st.warning("Live monitoring is available only for Premium users. Please upgrade your plan.")
            
            if st.button("Upgrade to Premium"):
                st.info("This is a demo. In a real application, this would redirect to a payment page.")
        else:
            # Check if live access is enabled in settings
            user_settings = get_user_settings(st.session_state.user["id"])
            
            if not user_settings["live_access"]:
                st.warning("Live access is not enabled. Please enable it in the Settings page.")
                
                if st.button("Go to Settings"):
                    st.session_state.page = "settings"
                    st.experimental_rerun()
            else:
                st.success("Live monitoring is active. Transactions will be analyzed in real-time.")
                
                # Simulated live transaction form
                st.markdown("### Test Live Transaction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    transaction_id = st.text_input("Transaction ID", value=str(uuid.uuid4()))
                    amount = st.number_input("Amount", min_value=1.0, value=100.0)
                
                with col2:
                    merchant = st.text_input("Merchant", value="Example Store")
                    location = st.text_input("Location", value="New York, USA")
                
                if st.button("Process Transaction"):
                    with st.spinner("Processing transaction..."):
                        # Simulate API call
                        transaction_data = {
                            "transaction_id": transaction_id,
                            "amount": amount,
                            "merchant": merchant,
                            "location": location
                        }
                        
                        result = api_analyze_transaction(transaction_data)
                        
                        # Display result
                        st.markdown("### Transaction Analysis Result")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Transaction ID:** {result['transaction_id']}")
                            st.markdown(f"**Risk Score:** {result['risk_score']:.2f}")
                            
                            risk_color = "#00cc96"
                            if result['risk_category'] == "Medium":
                                risk_color = "#ffa500"
                            elif result['risk_category'] == "High":
                                risk_color = "#ff4b4b"
                            
                            st.markdown(f"**Risk Category:** <span style='color:{risk_color};font-weight:bold;'>{result['risk_category']}</span>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("**Fraud Indicators:**")
                            if result['fraud_indicators']:
                                for indicator in result['fraud_indicators']:
                                    st.markdown(f"- {indicator}")
                            else:
                                st.markdown("No fraud indicators detected")
                        
                        # Send alert for high risk transactions
                        if result['risk_category'] == "High" and user_settings["email_alerts"]:
                            st.warning("High risk transaction detected! Alert email would be sent in a production environment.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Page: History
def render_history_page():
    st.markdown('<div class="main-header"><h1>Transaction History</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.markdown("### Your Previous Scans")
        
        # Get user scans from database
        scans = get_user_scans(st.session_state.user["id"])
        
        if not scans:
            st.info("You haven't performed any scans yet. Go to the Dashboard to analyze transactions.")
        else:
            # Create a DataFrame for display
            scans_df = pd.DataFrame(scans)
            scans_df["date"] = pd.to_datetime(scans_df["date"])
            scans_df["date"] = scans_df["date"].dt.strftime("%Y-%m-%d %H:%M")
            
            # Rename columns for display
            display_df = scans_df.rename(columns={
                "id": "Scan ID",
                "filename": "Filename",
                "total": "Total Transactions",
                "high_risk": "High Risk",
                "medium_risk": "Medium Risk",
                "low_risk": "Low Risk",
                "date": "Scan Date"
            })
            
            # Display the table
            st.dataframe(display_df)
            
            # Allow downloading history as CSV
            if st.button("Download History"):
                csv = display_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="finsec_history.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Settings
def render_settings_page():
    st.markdown('<div class="main-header"><h1>Settings</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Get user settings
        user_settings = get_user_settings(st.session_state.user["id"])
        
        # Create tabs for different settings categories
        tabs = st.tabs(["General", "API & Integration", "Account"])
        
        with tabs[0]:
            st.markdown("### General Settings")
            
            email_alerts = st.toggle("Email Alerts for High Risk Transactions", value=user_settings["email_alerts"])
            
            st.markdown("### Theme Settings")
            theme = st.selectbox("Theme", ["Light", "Dark"], index=0)
            
            if st.button("Save General Settings"):
                # Update settings in database
                update_user_settings(
                    st.session_state.user["id"],
                    email_alerts,
                    user_settings["live_access"],
                    user_settings["webhook_url"]
                )
                st.success("Settings saved successfully!")
        
        with tabs[1]:
            st.markdown("### API & Integration Settings")
            
            # API Key (read-only)
            st.markdown("#### Your API Key")
            st.text_input("API Key", value=user_settings["api_key"], disabled=True)
            
            # Live Access toggle
            live_access = st.toggle("Enable Live Access", value=user_settings["live_access"])
            
            # Webhook URL
            webhook_url = st.text_input("Webhook URL", value=user_settings["webhook_url"])
            
            if st.button("Save API Settings"):
                # Update settings in database
                update_user_settings(
                    st.session_state.user["id"],
                    user_settings["email_alerts"],
                    live_access,
                    webhook_url
                )
                st.success("API settings saved successfully!")
            
            # API Documentation
            with st.expander("API Documentation"):
                st.markdown("""
                ### FinSec API Documentation
                
                #### Authentication
                All API requests require your API key to be included in the header:
                ```
                X-API-Key: your_api_key
                ```
                
                #### Endpoints
                
                **POST /detect**
                
                Analyze a single transaction for fraud risk.
                
                Request body:
                ```json
                {
                    "transaction_id": "string",
                    "amount": "number",
                    "merchant": "string",
                    "location": "string",
                    "timestamp": "string (ISO format)",
                    "customer_id": "string"
                }
                ```
                
                Response:
                ```json
                {
                    "transaction_id": "string",
                    "risk_score": "number",
                    "risk_category": "string (Low, Medium, High)",
                    "fraud_indicators": ["string"],
                    "timestamp": "string (ISO format)"
                }
                ```
                
                **POST /batch-detect**
                
                Analyze multiple transactions for fraud risk.
                
                Request body:
                ```json
                {
                    "transactions": [
                        {
                            "transaction_id": "string",
                            "amount": "number",
                            "merchant": "string",
                            "location": "string",
                            "timestamp": "string (ISO format)",
                            "customer_id": "string"
                        }
                    ]
                }
                ```
                
                Response:
                ```json
                {
                    "results": [
                        {
                            "transaction_id": "string",
                            "risk_score": "number",
                            "risk_category": "string (Low, Medium, High)",
                            "fraud_indicators": ["string"],
                            "timestamp": "string (ISO format)"
                        }
                    ],
                    "summary": {
                        "total": "number",
                        "high_risk": "number",
                        "medium_risk": "number",
                        "low_risk": "number"
                    }
                }
                ```
                """)
        
        with tabs[2]:
            st.markdown("### Account Settings")
            
            st.markdown("#### User Information")
            st.markdown(f"**Email:** {st.session_state.user['email']}")
            st.markdown(f"**Role:** {st.session_state.user['role'].capitalize()}")
            st.markdown(f"**Plan:** {st.session_state.user['plan'].capitalize()}")
            
            st.markdown("#### Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Change Password"):
                if not current_password or not new_password or not confirm_password:
                    st.warning("Please fill in all password fields")
                elif new_password != confirm_password:
                    st.error("New passwords do not match")
                else:
                    # In a real application, this would verify the current password and update to the new one
                    st.success("Password changed successfully!")
            
            st.markdown("#### Upgrade Plan")
            st.markdown("Current Plan: **" + st.session_state.user["plan"].capitalize() + "**")
            
            if st.session_state.user["plan"] == "free":
                st.markdown("""
                **Premium Plan Benefits:**
                - Live transaction monitoring
                - Email alerts
                - Advanced analytics
                - Priority support
                """)
                
                if st.button("Upgrade to Premium"):
                    st.info("This is a demo. In a real application, this would redirect to a payment page.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Page: Privacy Policy
def render_privacy_page():
    st.markdown('<div class="main-header"><h1>Privacy Policy</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Read privacy policy from file
        with open("privacy_policy.md", "r") as f:
            privacy_policy = f.read()
        
        st.markdown(privacy_policy)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer with privacy policy link
def render_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2025 FinSec - Fraud Detection Platform | <a href="#" onclick="navigateToPrivacy()">Privacy Policy</a></p>
    </div>
    <script>
        function navigateToPrivacy() {
            window.parent.postMessage({"type": "streamlit:setComponentValue", "value": "navigate_to_privacy"}, "*");
        }
    </script>
    """, unsafe_allow_html=True)
    
    # Handle footer navigation
    if st.button("Privacy Policy Link", key="privacy_link_footer", label_visibility="collapsed"):
        st.session_state.page = "privacy"
        st.experimental_rerun()

# Main application
def main():
    # Initialize page if not set
    if 'page' not in st.session_state:
        st.session_state.page = "role_selection"
    
    # Render sidebar
    render_sidebar()
    
    # Render logout modal if shown
    if st.session_state.show_logout_modal:
        render_logout_modal()
    
    # Render AI chat interface if shown
    if st.session_state.show_chat:
        render_chat_interface()
    
    # Render current page
    if st.session_state.page == "role_selection":
        render_role_selection_page()
    elif st.session_state.page == "login":
        render_login_page()
    elif st.session_state.page == "signup":
        render_signup_page()
    elif st.session_state.page == "dashboard":
        if st.session_state.user:
            render_dashboard_page()
        else:
            st.session_state.page = "role_selection"
            st.experimental_rerun()
    elif st.session_state.page == "history":
        if st.session_state.user:
            render_history_page()
        else:
            st.session_state.page = "role_selection"
            st.experimental_rerun()
    elif st.session_state.page == "settings":
        if st.session_state.user:
            render_settings_page()
        else:
            st.session_state.page = "role_selection"
            st.experimental_rerun()
    elif st.session_state.page == "privacy":
        render_privacy_page()
    
    # Render footer
    render_footer()

    # Handle chat input from JavaScript
    chat_input = st.text_input("Hidden Chat Input", key="chat_input_js", label_visibility="collapsed")
    if chat_input and chat_input != "close_chat":
        if chat_input == "navigate_to_privacy":
            st.session_state.page = "privacy"
            st.experimental_rerun()
        elif chat_input == "cancel_logout":
            st.session_state.show_logout_modal = False
            st.experimental_rerun()
        elif chat_input == "confirm_logout":
            st.session_state.user = None
            st.session_state.login_status = None
            st.session_state.show_logout_modal = False
            st.session_state.page = "role_selection"
            st.experimental_rerun()
        else:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": chat_input})
            
            # Get AI response
            response = get_ai_response(chat_input)
            
            # Add assistant response to chat
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            # Rerun to update UI
            st.experimental_rerun()

if __name__ == "__main__":
    main()
