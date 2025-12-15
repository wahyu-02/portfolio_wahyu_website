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
    "temperature": 0.3,       # Tetap rendah agar faktual
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1000, # Sedikit diperpanjang untuk jawaban detail
    "response_mime_type": "text/plain",
}

# Safety Settings (Agar tidak over-sensitive memblokir konten teknis)
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

# --- KNOWLEDGE BASE (Diperbarui dari CV & HTML) ---
CONTEXT_DATA = """
[PROFIL UTAMA]
Nama: Wahyudiyanto (Wade).
Email: lajazwade@gmail.com
Kontak: +62 853 9844 8838
Tautan: linkedin.com/in/wahyudiyanto | github.com/wahyu-02 | wahyudata.web.id
Lokasi: Indonesia.

[RINGKASAN PROFESIONAL]
Data Analyst yang berorientasi pada detail dengan spesialisasi di Blockchain Analytics dan Financial Modeling.
Ahli dalam menerjemahkan dataset kompleks menjadi keputusan bisnis menggunakan Python (Pandas/NumPy) dan SQL.
Berpengalaman membangun dashboard interaktif (Looker Studio, Tableau) dan analisis statistik (Volatility Modeling).

[PENDIDIKAN]
Institut Teknologi Sumatera (ITERA) - S1 Data Science (Agustus 2021 - Agustus 2025).
GPA: 3.28/4.00.

[PENGALAMAN KERJA]
1. Big Data Analysis Practicum Assistant - ITERA (Feb 2025 - Jul 2025):
   - Mentoring mahasiswa dalam optimasi query SQL dan scripting Python.
   - Debugging kode kompleks untuk pemrosesan data skala besar.
2. Head of Creative Media (Data Analytics) - HMSD (Feb 2024 - Nov 2024):
   - Analisis kuantitatif KPI media sosial untuk strategi konten.
   - Memimpin tim beranggotakan 19 orang.

[SKILL TEKNIS]
- Crypto & Web3: On-chain Data Analysis, Liquidity Tracking, Fear & Greed Index, Market Sentiment.
- Visualisasi: Looker Studio, Tableau, Streamlit, Matplotlib, Seaborn.
- Statistik: ARCH/GARCH Modeling, Time-Series Forecasting, Regression, A/B Testing.
- Core Stack: Python (Pandas, NumPy, Scikit-learn), SQL (PostgreSQL, MySQL), Google Cloud Platform.
- Bahasa: Inggris (Advanced C1 - EF SET 67/100), Indonesia (Native).

[DAFTAR PROYEK UNGGULAN (FEATURED PROJECTS)]
1. **XandeumNexus: Crypto Network Analytics (2025)**
   - *Tech:* On-chain Data, SQL, Python.
   - *Deskripsi:* Engine analitik live untuk memonitor kesehatan jaringan blockchain (Paging Efficiency & Latency) dan dashboard performa node.
   - *Link:* github.com/0xlajaz/xandeum-nexus

2. **Bitcoin Price Prediction / Thesis (2025)**
   - *Tech:* LSTM, Crypto Fear & Greed Index, Python.
   - *Deskripsi:* Model Multivariate LSTM untuk prediksi harga Bitcoin dengan error rendah (MAPE 2.33%). Mengalahkan model baseline standar.
   - *Link:* github.com/wahyu-02/Bitcoin-FGI-LSTM

3. **Global Energy Trends Dashboard (2024)**
   - *Tech:* Looker Studio.
   - *Deskripsi:* Dashboard interaktif untuk menganalisis tren adopsi energi terbarukan global.
   - *Link:* lookerstudio.google.com/reporting/e9059d91-2b77-43a8-b109-a010435aa31e

4. **ARCH & GARCH Volatility Modeling (2024)**
   - *Tech:* R, Statistical Analysis.
   - *Deskripsi:* Pemodelan volatilitas time-series untuk data iklim dan keuangan.
   - *Link:* rpubs.com/Wahyudiyanto02/1151156

5. **Walmart Sales XGBoost Prediction**
   - *Tech:* XGBoost, Streamlit.
   - *Deskripsi:* Dashboard prediksi penjualan mingguan Walmart.
   - *Link:* walmart-sales-prediction-eydj6qjxmxfcjtyhck8unm.streamlit.app

6. **Bioactivity Prediction (Breast Cancer Therapy)**
   - *Tech:* Machine Learning (Random Forest).
   - *Deskripsi:* Prediksi aktivitas senyawa herbal sebagai inhibitor HDAC.
   - *Link:* github.com/wahyu-02/Potensi-Senyawa-Herbal-sebagai-Inhibitor-HDAC-untuk-Terapi-Kanker-Payudara

7. **Big Data Analysis (Amazon Product)**
   - *Tech:* Hadoop, Spark, Python.
   - *Link:* github.com/wahyu-02/Big-Data-Analysis_Amazon-Product

8. **CLIP Model Evaluation (Livestock Classification)**
   - *Tech:* Computer Vision, Zero-shot learning.
   - *Link:* github.com/sains-data/EVALUASI-PERFORMA-MODEL-CLIP-DALAM-KLASIFIKASI-GAMBAR-TEKS-PADA-TERNAK

9. **Customer Segmentation**
   - *Tech:* K-Means Clustering.
   - *Deskripsi:* Segmentasi pelanggan untuk strategi marketing.

10. **Monte Carlo Stock Simulation**
    - *Tech:* R, Stochastic Modeling.
    - *Link:* api.rpubs.com/Wahyudiyanto02/1150953
"""

# --- SYSTEM INSTRUCTION (Persona & Logic) ---
SYSTEM_PROMPT = f"""
You are **Wade's AI Assistant**, an intelligent portfolio chatbot for Wahyudiyanto (Wade).
Your goal is to impress recruiters and visitors by providing accurate, professional, and enthusiastic answers about Wade's profile.

**CORE KNOWLEDGE:**
{CONTEXT_DATA}

**GUIDELINES:**
1. **Language & Tone:** - STRICTLY answer in the **same language** as the user (Indonesian or English).
   - Tone: Professional, confident, yet friendly. Showcase Wade's expertise subtly.
2. **Context Awareness:** - ONLY answer questions related to Wade (Skills, Projects, Experience, Contact).
   - If asked about general topics (e.g., "What is the capital of France?"), politely refuse: "I specialize in discussing Wahyu's professional background. Shall we talk about his Data Science projects?"
3. **Detail Level:** - Be concise (2-3 sentences) for simple questions.
   - Use **bullet points** for lists (skills, projects).
   - ALWAYS provide the **Project Link** if a specific project is discussed.
4. **Call to Action:** - Occasionally suggest contacting him directly via email (lajazwade@gmail.com) for collaboration opportunities.

**Behavior Examples:**
- User: "What is his best project?" -> Answer: Highlight 'Bitcoin Price Prediction' or 'XandeumNexus' and explain why it's impressive (low MAPE, real-time analytics).
- User: "Bisa hubungi dia?" -> Answer: "Tentu! Anda bisa menghubungi Wahyu via email di lajazwade@gmail.com atau LinkedIn."
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

        # Start Chat Session with History (Stateless but Contextual)
        # Mengirim System Prompt di awal setiap request memastikan konteks selalu terbawa
        chat_session = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I am ready to represent Wahyudiyanto (Wade) professionally."]}
        ])
        
        response = chat_session.send_message(user_message)
        ai_reply = response.text.strip()
        
        logger.info(f"🤖 AI replied: {ai_reply}")

        return jsonify({"reply": ai_reply})

    except Exception as e:
        logger.error(f"🔥 Error processing request: {str(e)}")
        return jsonify({
            "reply": "Maaf, terjadi kesalahan pada sistem AI saya. Silakan hubungi Wahyu secara langsung melalui Email atau LinkedIn."
        }), 500

if __name__ == "__main__":
    # Port diambil dari Environment Variable (Wajib untuk Cloud Run / Render)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
