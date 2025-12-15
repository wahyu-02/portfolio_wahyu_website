import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Inisialisasi Aplikasi ---
app = Flask(__name__)

# Konfigurasi CORS (Penting untuk akses dari GitHub Pages)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Gemini AI ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("⚠️ WARNING: GEMINI_API_KEY belum diset!")

genai.configure(api_key=API_KEY)

# Konfigurasi Model
generation_config = {
    "temperature": 0.4,       # Tetap rendah agar faktual
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000, 
    "response_mime_type": "text/plain",
}

# Safety Settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Inisialisasi Model 
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- KNOWLEDGE BASE (Pure English for Consistency) ---
CONTEXT_DATA = """
[PROFILE]
Name: Wahyu (Wade).
Email: lajazwade@gmail.com

[PROFESSIONAL SUMMARY]
Detail-oriented Data Analyst specializing in Blockchain Analytics and Financial Modeling.
Expert in interpreting complex datasets to drive decision-making using Python (Pandas/NumPy) and SQL.
Experienced in building interactive dashboards (Looker Studio, Tableau) and statistical analysis (Volatility Modeling).

[EDUCATION]
Institut Teknologi Sumatera (ITERA) - Bachelor in Data Science (Aug 2021 - Aug 2025).
GPA: 3.28/4.00.

[EXPERIENCE]
1. Big Data Analysis Practicum Assistant - ITERA (Feb 2025 - Jul 2025):
   - Mentored students in SQL query optimization and Python scripting.
   - Debugged complex code for large-scale data processing.
2. Head of Creative Media (Data Analytics) - HMSD (Feb 2024 - Nov 2024):
   - Conducted quantitative analysis of social media KPIs.
   - Led a team of 19 members.

[TECHNICAL SKILLS]
- Crypto & Web3: On-chain Data Analysis, Liquidity Tracking, Fear & Greed Index, Market Sentiment.
- Visualization: Looker Studio, Tableau, Streamlit, Matplotlib, Seaborn.
- Statistics: ARCH/GARCH Modeling, Time-Series Forecasting, Regression, A/B Testing.
- Core Stack: Python (Pandas, NumPy, Scikit-learn), SQL (PostgreSQL, MySQL), Google Cloud Platform.
- Languages: English (Advanced C1 - EF SET 67/100), Indonesian (Native).

[FEATURED PROJECTS]
1. XandeumNexus: Crypto Network Analytics (2025)
   - Tech: On-chain Data, SQL, Python.
   - Description: Live analytics engine to monitor blockchain network health (Paging Efficiency & Latency).
   - Link: github.com/0xlajaz/xandeum-nexus

2. Bitcoin Price Prediction / Thesis (2025)
   - Tech: LSTM, Crypto Fear & Greed Index, Python.
   - Description: Multivariate LSTM model for Bitcoin price prediction with low error (MAPE 2.33%).
   - Link: github.com/wahyu-02/Bitcoin-FGI-LSTM

3. Global Energy Trends Dashboard (2024)
   - Tech: Looker Studio.
   - Description: Interactive dashboard analyzing global renewable energy adoption trends.
   - Link: lookerstudio.google.com/reporting/e9059d91-2b77-43a8-b109-a010435aa31e

4. ARCH & GARCH Volatility Modeling (2024)
   - Tech: R, Statistical Analysis.
   - Description: Time-series volatility modeling for climate and financial data.
   - Link: rpubs.com/Wahyudiyanto02/1151156

5. Walmart Sales XGBoost Prediction
   - Tech: XGBoost, Streamlit.
   - Description: Weekly sales prediction dashboard for Walmart.
   - Link: walmart-sales-prediction-eydj6qjxmxfcjtyhck8unm.streamlit.app

6. Bioactivity Prediction (Breast Cancer Therapy)
   - Tech: Machine Learning (Random Forest).
   - Link: github.com/wahyu-02/Potensi-Senyawa-Herbal-sebagai-Inhibitor-HDAC-untuk-Terapi-Kanker-Payudara

7. Big Data Analysis (Amazon Product)
   - Tech: Hadoop, Spark, Python.
   - Link: github.com/wahyu-02/Big-Data-Analysis_Amazon-Product

8. CLIP Model Evaluation (Livestock Classification)
   - Tech: Computer Vision, Zero-shot learning.
   - Link: github.com/sains-data/EVALUASI-PERFORMA-MODEL-CLIP-DALAM-KLASIFIKASI-GAMBAR-TEKS-PADA-TERNAK

9. Customer Segmentation
   - Tech: K-Means Clustering.
   - Description: Customer segmentation for marketing strategy.

10. Monte Carlo Stock Simulation
    - Tech: R, Stochastic Modeling.
    - Link: api.rpubs.com/Wahyudiyanto02/1150953
"""

# --- SYSTEM INSTRUCTION (Persona & Logic) ---
SYSTEM_PROMPT = f"""
You are **Wade's AI Assistant**, an intelligent portfolio chatbot for Wahyudiyanto (Wade).

**CORE KNOWLEDGE:**
{CONTEXT_DATA}

**STRICT GUIDELINES:**
1. **LANGUAGE:** **ALWAYS** answer in **ENGLISH**. Even if the user asks in Indonesian, you MUST reply in English.
2. **FORMATTING:** - Keep it clean and professional. 
   - Use simple bullet points (•) for lists. 
   - **DO NOT** use messy nesting like `* **Project**`. 
   - Format projects like this: 
     • **Project Name**: Description (Tech Stack). [Link]
3. **SCOPE:** Answer ONLY about Wade's professional profile (Skills, Projects, Experience).
4. **IDENTITY:** You are an AI assistant. Refer to Wahyu as "he" or "Wahyu".
5. **LENGTH:** Keep answers concise (2-4 sentences) unless asked for a full list.
"""

# --- Routes ---

@app.route("/")
def home():
    return jsonify({
        "status": "active", 
        "service": "Wade Portfolio AI Backend", 
        "model": "gemini-1.5-flash"
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

        chat_session = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I will always answer in English and use clean formatting."]}
        ])
        
        response = chat_session.send_message(user_message)
        ai_reply = response.text.strip()
        
        logger.info(f"🤖 AI replied: {ai_reply}")

        return jsonify({"reply": ai_reply})

    except Exception as e:
        logger.error(f"🔥 Error processing request: {str(e)}")
        return jsonify({
            "reply": "I apologize, but I am currently experiencing technical difficulties. Please try again later or contact Wahyu directly via Email."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
