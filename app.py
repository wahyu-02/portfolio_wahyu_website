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

# Konfigurasi CORS
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
    "temperature": 0.5,
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

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- KNOWLEDGE BASE ---
CONTEXT_DATA = """
[PROFILE]
Name: Wahyudiyanto (Wade).
Email: lajazwade@gmail.com
Contact: +62 853 9844 8838
Links: linkedin.com/in/wahyudiyanto | github.com/wahyu-02 | wahyudata.web.id
Location: Indonesia.

[PROFESSIONAL SUMMARY]
Detail-oriented Data Analyst specializing in Blockchain Analytics and Financial Modeling.
Expert in interpreting complex datasets using Python (Pandas/NumPy) and SQL.
Experienced in building interactive dashboards (Looker Studio, Tableau) and statistical analysis (Volatility Modeling).

[FEATURED PROJECTS]
1. XandeumNexus: Crypto Network Analytics (2025)
   - Tech: On-chain Data, SQL, Python.
   - Description: Live analytics engine to monitor blockchain network health (Paging Efficiency & Latency).
   - Link: https://github.com/0xlajaz/xandeum-nexus

2. Bitcoin Price Prediction / Thesis (2025)
   - Tech: LSTM, Crypto Fear & Greed Index, Python.
   - Description: Multivariate LSTM model for Bitcoin price prediction with low error (MAPE 2.33%).
   - Link: https://github.com/wahyu-02/Bitcoin-FGI-LSTM

3. Global Energy Trends Dashboard (2024)
   - Tech: Looker Studio.
   - Description: Interactive dashboard analyzing global renewable energy adoption trends.
   - Link: https://lookerstudio.google.com/reporting/e9059d91-2b77-43a8-b109-a010435aa31e

4. ARCH & GARCH Volatility Modeling (2024)
   - Tech: R, Statistical Analysis.
   - Description: Time-series volatility modeling for climate and financial data.
   - Link: https://rpubs.com/Wahyudiyanto02/1151156

5. Walmart Sales XGBoost Prediction
   - Tech: XGBoost, Streamlit.
   - Description: Weekly sales prediction dashboard for Walmart.
   - Link: https://walmart-sales-prediction-eydj6qjxmxfcjtyhck8unm.streamlit.app

6. Bioactivity Prediction (Breast Cancer Therapy)
   - Tech: Machine Learning (Random Forest).
   - Link: https://github.com/wahyu-02/Potensi-Senyawa-Herbal-sebagai-Inhibitor-HDAC-untuk-Terapi-Kanker-Payudara

7. Big Data Analysis (Amazon Product)
   - Tech: Hadoop, Spark, Python.
   - Link: https://github.com/wahyu-02/Big-Data-Analysis_Amazon-Product

8. CLIP Model Evaluation (Livestock Classification)
   - Tech: Computer Vision, Zero-shot learning.
   - Link: https://github.com/sains-data/EVALUASI-PERFORMA-MODEL-CLIP-DALAM-KLASIFIKASI-GAMBAR-TEKS-PADA-TERNAK

9. Customer Segmentation
   - Tech: K-Means Clustering.
   - Description: Customer segmentation for marketing strategy.

10. Monte Carlo Stock Simulation
    - Tech: R, Stochastic Modeling.
    - Link: https://api.rpubs.com/Wahyudiyanto02/1150953
"""

# --- SYSTEM INSTRUCTION (Updated for HTML Output) ---
SYSTEM_PROMPT = f"""
You are **Wade's AI Assistant**, an intelligent portfolio chatbot for Wahyudiyanto (Wade).

**CORE KNOWLEDGE:**
{CONTEXT_DATA}

**STRICT GUIDELINES:**
1. **LANGUAGE:** ALWAYS answer in **ENGLISH**.
2. **FORMATTING (CRITICAL):**
   - The user's interface renders **HTML**. You MUST use HTML tags to structure your answer.
   - **DO NOT** use Markdown (like `**bold**` or `[link](url)`).
   - Use `<b>` for bold text.
   - Use `<br>` for line breaks.
   - Use `<ul>` and `<li>` for lists to keep them tidy.
   - **LINKS:** MUST be formatted as: 
     `<a href="URL_HERE" target="_blank" style="color: #2dd4bf; text-decoration: underline;">See Project</a>` 
     (This ensures they are clickable and visible).

3. **RESPONSE TEMPLATE FOR PROJECTS:**
   When listing projects, use this exact HTML structure:
   <ul>
     <li><b>Project Name</b>: Short description. <a href="LINK" target="_blank" style="color: #2dd4bf;">[View Code]</a></li>
   </ul>

4. **IDENTITY:** Answer ONLY about Wade. Keep it professional and concise.
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
            {"role": "model", "parts": ["Understood. I will answer in English using HTML tags (<ul>, <li>, <b>, <a href>) so the output is neat and links are clickable."]}
        ])
        
        response = chat_session.send_message(user_message)
        ai_reply = response.text.strip()
        
        logger.info(f"🤖 AI replied: {ai_reply}")

        return jsonify({"reply": ai_reply})

    except Exception as e:
        logger.error(f"🔥 Error processing request: {str(e)}")
        return jsonify({
            "reply": "I apologize, but I am currently experiencing technical difficulties. Please try again later."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
