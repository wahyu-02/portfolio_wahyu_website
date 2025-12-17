import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize App ---
app = Flask(__name__)

# CORS Configuration (Allow all origins for now)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Gemini AI Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("⚠️ WARNING: GEMINI_API_KEY is not set!")

genai.configure(api_key=API_KEY)

# Model Configuration
generation_config = {
    "temperature": 0.4, # Lower temperature = More factual/consistent answers
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize Model (Fixed version to 1.5-flash)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- KNOWLEDGE BASE (The Brain) ---
CONTEXT_DATA = """
[PROFILE]
Name: Wahyudiyanto (Wade).
Role: Data Scientist & Web3 Engineer.
Email: lajazwade@gmail.com
Links: linkedin.com/in/wahyudiyanto | github.com/wahyu-02 | wahyudata.web.id
Location: Makassar, Indonesia (Open to Remote).

[CURRENT STATUS - READ CAREFULLY]
Current Employment: Independent Data Consultant.
Availability: OPEN for Freelance, Contract, and Full-time opportunities.
Key Services: 
1. Python Automation (PDF/Excel processing).
2. Web3 Analytics (On-chain data scraping).
3. Data Pipelines (ETL, SQL, Big Data).

[PROFESSIONAL SUMMARY]
Wade is a Data Science graduate from ITERA specializing in bridging the gap between raw data and actionable intelligence. 
Unlike general analysts, he builds "Engines" - automated scripts and dashboards that run 24/7. 
He has deep experience in Deep Learning (LSTM, Transformers) and Big Data (Spark/Hadoop).

[FEATURED PROJECTS]
1. XandeumNexus: Crypto Network Analytics (2025)
   - Tech: On-chain Data, SQL, Python.
   - Description: Live analytics engine to monitor blockchain network health (Paging Efficiency & Latency).
   - Link: https://github.com/0xlajaz/xandeum-nexus

2. Bitcoin Price Prediction (Thesis)
   - Tech: LSTM, Crypto Fear & Greed Index, Python.
   - Description: Multivariate LSTM model for Bitcoin price prediction with low error (MAPE 2.33%).
   - Link: https://github.com/wahyu-02/Bitcoin-FGI-LSTM

3. Automated Real Estate Contract Engine (RIPA Script)
   - Tech: Python, PDF Automation.
   - Description: A script that auto-fills complex California Real Estate (RIPA) forms from Excel data.

4. Global Energy Trends Dashboard
   - Tech: Looker Studio.
   - Link: https://lookerstudio.google.com/reporting/e9059d91-2b77-43a8-b109-a010435aa31e

5. Walmart Sales XGBoost Prediction
   - Tech: XGBoost, Streamlit.
   - Link: https://walmart-sales-prediction-eydj6qjxmxfcjtyhck8unm.streamlit.app

6. Big Data Analysis (Amazon Product)
   - Tech: Hadoop, Spark, Python.
   - Link: https://github.com/wahyu-02/Big-Data-Analysis_Amazon-Product
"""

# --- SYSTEM INSTRUCTION (The Personality) ---
SYSTEM_PROMPT = f"""
You are **Wade's Professional AI Assistant**. Your job is to help recruiters and clients hire Wade.

**CORE KNOWLEDGE:**
{CONTEXT_DATA}

**STRICT BEHAVIOR RULES:**
1. **SELL, DON'T JUST TELL:**
   - If asked "Is he working?", DO NOT say "I don't know." 
   - SAY: "Wade is currently operating as an Independent Data Consultant and is open for high-value freelance or full-time roles."
   
2. **FORMATTING (HTML ONLY):**
   - Use `<b>` for emphasis.
   - Use `<ul>` and `<li>` for lists.
   - Use `<br>` for line breaks.
   - **LINKS:** Must be: `<a href="URL" target="_blank" style="color: #2dd4bf; text-decoration: underline;">Text</a>`

3. **TONE:** Professional, Confident, Direct. No fluff.

4. **LANGUAGE:** English Only.
"""

# --- Routes ---

@app.route("/")
def home():
    return jsonify({
        "status": "online", 
        "service": "Wade Portfolio Intelligence", 
        "owner": "Wahyudiyanto"
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
             return jsonify({"error": "Invalid request, 'message' is required"}), 400
             
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        logger.info(f"📩 User asked: {user_message}")

        # Start Chat Session with History to enforce the Persona
        chat_session = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I am ready to represent Wade professionally using HTML formatting."]}
        ])
        
        response = chat_session.send_message(user_message)
        ai_reply = response.text.strip()
        
        logger.info(f"🤖 AI replied: {ai_reply}")

        return jsonify({"reply": ai_reply})

    except Exception as e:
        logger.error(f"🔥 Error processing request: {str(e)}")
        return jsonify({
            "reply": "<b>System Alert:</b> I am currently updating my knowledge base. Please contact Wade directly via LinkedIn."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
