from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import logging

# --- Inisialisasi Aplikasi ---
app = Flask(__name__)

# Konfigurasi CORS (Izinkan semua origin untuk endpoint chatbot)
CORS(app, resources={r"/*": {"origins": "*"}})

# Setup logging agar mudah debugging di dashboard Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Gemini AI ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY belum diset di environment variables!")

genai.configure(api_key=API_KEY)

# Konfigurasi model untuk respon yang lebih faktual dan konsisten
generation_config = {
    "temperature": 0.3,  # Rendah agar jawaban tidak ngawur/kreatif berlebihan
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500, # Batasi panjang jawaban agar ringkas
    "response_mime_type": "text/plain",
}

# Gunakan model flash karena lebih cepat untuk chatbot real-time
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    generation_config=generation_config
)

# --- Data Konteks (Knowledge Base) ---
# Kita gabungkan konteks agar AI punya pemahaman utuh. 
# AI Gemini sangat pintar mendeteksi bahasa, jadi kita tidak perlu library eksternal.

CONTEXT_DATA = """
[INFORMASI BAHASA INDONESIA]
Nama: Wahyu (Wade).
Role: Data Scientist / Data Analyst.
Pendidikan: Institut Teknologi Sumatera (ITERA), fokus Data Science.
Motivasi: Terinspirasi Jack Ma "Data is the new oil". Ingin jadi bagian revolusi digital.
Minat: Analisis data, ML, visualisasi data, coding, investasi (mengagumi Timothy Ronald), blockchain, AI.
Hobi: Nonton film/drama/donghua, baca jurnal.
Skill Teknis: Python (Pandas, NumPy, Scikit-Learn), SQL, R, MongoDB, Tableau, Looker, Keras (Deep Learning), Cloud Computing (dasar), Big Data (Hadoop/Spark).
Tools: Jupyter Notebook, VS Code, ChatGPT.
Pengalaman Proyek:
1. Evaluasi Model CLIP (Klasifikasi ternak, akurasi 85%).
2. Automated Data Backup System.
3. Predictive Maintenance Dashboard (Random Forest, akurasi 100%).
4. Monte Carlo Stock Price Simulation.
5. Analisis Data Keuangan & Eksplorasi Data Industri.
6. Analisis Big Data (Market trends).
7. Customer Segmentation (K-Means).
8. Prediksi Bioaktivitas Senyawa Herbal (Kanker Payudara).
Organisasi: Head of Creative Media di HMSD (Data Science Student Association).
Pengalaman Mengajar: Asisten Lab Algoritma & Pemrograman, Asisten Lab Struktur Data, Volunteer ITERA Mengajar.
Prestasi:
- Juara 1 Videography & Best Presentation (Pertamina Workers Union).
- Juara 2 Lomba Video Halo China (FPCI & Kedubes China).
- Penerima Beasiswa BriLian (BRI).
- Sertifikat Google Advanced Data Analytics.
Tujuan Karir: Mencari magang/kerja di industri data, ingin jadi master AI & Blockchain.
Pesan: "Jangan setengah-setengah! Kuasai skill ini karena sangat berguna di masa depan."

[ENGLISH INFORMATION]
Name: Wahyu (Wade).
Role: Data Scientist / Data Analyst.
Education: ITERA (Sumatra Institute of Technology), Data Science focus.
Inspiration: Jack Ma ("Data is the new oil").
Interests: Data analysis, ML, Viz, Coding, Investment, Blockchain, AI.
Tech Stack: Python, SQL, R, MongoDB, Tableau, Looker, Keras, Basic Cloud/Big Data.
Projects: CLIP Model Evaluation, Automated Backup, Predictive Maintenance (ML), Monte Carlo Simulation, Financial Analysis, Big Data Analysis, Customer Segmentation, Bioactivity Prediction.
Experience: Head of Creative Media (HMSD), Lab Assistant (Algo & Data Structures), Volunteer (ITERA Mengajar).
Achievements: 1st Place Videography (Pertamina), 2nd Place Halo China Video, BriLian Scholarship, Google Advanced Data Analytics Certificate.
Goal: Seeking internship/job in data industry.
"""

# --- System Instruction (Persona) ---
SYSTEM_PROMPT = f"""
You are Wade's AI Assistant (Portfolio Chatbot).
Your job is to answer questions about Wahyu (Wade) based ONLY on the context provided below.

CONTEXT:
{CONTEXT_DATA}

INSTRUCTIONS:
1. **Language:** Always answer in the **SAME LANGUAGE** as the user's question. If they ask in English, answer in English. If Indonesian, answer in Indonesian.
2. **Tone:** Professional, enthusiastic, friendly, and concise.
3. **Identity:** Refer to Wahyu as "he" or "Wahyu". You are his assistant.
4. **Scope:** If the user asks something NOT in the context (e.g., "What is the weather?"), politely apologize and say you only know about Wahyu's professional profile.
5. **Format:** Keep answers short (2-4 sentences) unless asked for details. Use formatting (bullet points) if listing skills or projects.
"""

# --- Routes ---

@app.route("/")
def home():
    return jsonify({"status": "Chatbot API is Running", "model": "gemini-1.5-flash"})

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "Invalid JSON"}), 400
             
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"📩 User asked: {user_message}")

        # --- Logic Chatbot ---
        # Kita kirim System Prompt + User Message ke Gemini
        # Gemini 1.5 Flash memiliki context window besar, jadi ini sangat efisien.
        
        chat = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Understood. I am Wade's AI Assistant. I will answer questions about him based on the provided context in the user's language."]}
        ])
        
        response = chat.send_message(user_message)
        ai_reply = response.text.strip()
        
        logger.info(f"🤖 AI replied: {ai_reply}")

        return jsonify({"response": ai_reply})

    except Exception as e:
        logger.error(f"🔥 Error: {e}")
        # Fallback response yang aman
        return jsonify({
            "response": "Maaf, Wade AI sedang mengalami gangguan koneksi. Silakan coba lagi nanti atau hubungi Wahyu langsung via email."
        }), 500

# Konfigurasi untuk Gunicorn/Render
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
