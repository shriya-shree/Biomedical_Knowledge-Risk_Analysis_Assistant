import streamlit as st
import faiss
import pickle
import numpy as np
import joblib
import google.generativeai as genai
import db
import re

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)
bert_model.eval()
# ================= CONFIG =================
GEMINI_API_KEY = "AIzaSyD7mv8HU-fhck8eTvp4JtH6-WdeZ9Mx7ZA"
# EMBED_MODEL = "models/text-embedding-004"
GEN_MODEL = "models/gemini-flash-latest"
TOP_K = 3
# =========================================

# -------- Gemini Setup --------
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)

# -------- Streamlit Config --------
st.set_page_config(page_title="Biomedical RAG Assistant")

# -------- Load RAG Assets --------
index = faiss.read_index("pubmed_faiss.index")
with open("pubmed_texts.pkl", "rb") as f:
    texts = pickle.load(f)

# -------- Load Risk Model --------
risk_model = joblib.load("risk_model/diabetes_risk_model.pkl")

feature_names = [
    "HighBP","HighChol","BMI","Smoker","Stroke","HeartDisease",
    "PhysActivity","Fruits","Veggies","Age","Education","Income",
    "GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","CholCheck",
    "AnyHealthcare","NoDocbcCost","HvyAlcoholConsump"
]

importances = risk_model.feature_importances_

if "page" not in st.session_state:
    st.session_state.page = "chat"

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None

SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm",
    "cut myself", "die", "overdose", "hang myself",
    "worthless", "no reason to live"
]

def is_self_harm_query(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in SELF_HARM_KEYWORDS)

def safe_self_harm_response():
    return (
        "I'm really sorry that you're feeling this way. "
        "I’m not able to help with anything that could harm you.\n\n"
        "You’re not alone, and support is available. "
        "If you’re feeling overwhelmed or in danger, please consider reaching out "
        "to a trusted person or a mental health professional.\n\n"
        "If you are in India, you can call **AASRA: 91-9820466726**.\n"
        "If you are elsewhere, local emergency services or a suicide prevention hotline "
        "can provide immediate support."
    )

def alert_admin(user_id, message):
    db.cursor.execute(
        "INSERT INTO safety_alerts (user_id, message) VALUES (?, ?)",
        (user_id, message)
    )
    db.conn.commit()

# ================= RAG FUNCTIONS =================

def embed_query(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().numpy()

def retrieve(q: str):
    q_emb = np.array([embed_query(q)]).astype("float32")
    _, idx = index.search(q_emb, TOP_K)
    return idx[0]


def answer(q: str):
    
    db.cursor.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp ASC",
        (st.session_state.active_chat_id,)
    )
    past_msgs = db.cursor.fetchall()

    history = "\n".join([f"{role}: {content}" for role, content in past_msgs[-6:]])

    doc_ids = retrieve(q)
    context = "\n\n".join([texts[i] for i in doc_ids])

    prompt = f"""
You are a biomedical knowledge and risk analysis assistant.

Answer the user's question using ONLY the biomedical evidence
present in the context below.

Structure your response as:
1. Explanation
2. Evidence summary
3. Limitation of evidence

Conversation so far:
{history}

Context:
{context}

Question:
{q}
"""
    response = gen_model.generate_content(prompt)
    pmids = [extract_pmid(texts[i]) for i in doc_ids]
    return response.text, pmids

def generate_chat_title(question: str):
    prompt = f"""
Generate a short 4–6 word title for this medical query.
Do NOT use punctuation.

Query:
{question}
"""
    response = gen_model.generate_content(prompt)
    return response.text.strip()

def risk_guidance(risk_level: str):
    prompt = f"""
You are a biomedical assistant.

Given that the user has a {risk_level} overall health risk,
provide evidence-based lifestyle and medical precautions.
Keep it concise and practical.
"""
    response = gen_model.generate_content(prompt)
    return response.text

def extract_pmid(text):
    match = re.search(r"PMID[:\s]+(\d+)", text)
    return match.group(1) if match else "Unknown PMID"


# ================= RISK FUNCTIONS =================

def explain_risk(features, top_n=3):
    contrib = list(zip(feature_names, importances, features))
    active = [c for c in contrib if c[2] > 0]
    active_sorted = sorted(active, key=lambda x: x[1], reverse=True)
    return [name for name, _, _ in active_sorted[:top_n]]


def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            db.cursor.execute(
                "SELECT user_id FROM users WHERE username=? AND password=?",
                (username, password)
            )
            user = db.cursor.fetchone()

            if user:
                st.session_state.user_id = user[0]
                st.session_state.page = "chat"
                st.rerun()
            else:
                st.warning("User not found. Please register first.")

        except Exception as e:
            st.error("Login system not initialized. Please restart the app.")

    if st.button("Register"):
        try:
            db.cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            db.conn.commit()
            st.success("Registration successful. Please login.")
        except:
            st.error("Username already exists.")


# -------- Login Gate --------
if not st.session_state.user_id:
    login_page()
    st.stop()

# ================= UI =================

if st.session_state.page == "chat":

    st.title("Biomedical Knowledge Assistant")

    if not st.session_state.active_chat_id:
        st.info("Start a new conversation from the sidebar.")
    else:
        db.cursor.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY timestamp",
            (st.session_state.active_chat_id,)
        )
        messages = db.cursor.fetchall()

        for role, content in messages:
            st.chat_message(role).write(content)

        user_input = st.chat_input("Ask any medical question...")

        if user_input:

            # 🚨 SAFETY CHECK
            if is_self_harm_query(user_input):
                alert_admin(
                    st.session_state.user_id,
                    user_input
                )

                with st.chat_message("assistant"):
                    st.error("⚠️ Sensitive Topic Detected")
                    st.write(safe_self_harm_response())

                # Save message (optional, but recommended)
                db.cursor.execute(
                    "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                    (st.session_state.active_chat_id, "assistant", "[Self-harm safety response triggered]")
                )
                db.conn.commit()

                st.stop()  # ⛔ Prevent LLM call
            db.cursor.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (st.session_state.active_chat_id, "user", user_input)
            )
            db.conn.commit()

            # Set title on first user message
            if len(messages) == 0:
                title = generate_chat_title(user_input)
                db.cursor.execute(
                    "UPDATE chats SET title=? WHERE chat_id=?",
                    (title, st.session_state.active_chat_id)
                )
                db.conn.commit()


            answer_text, sources = answer(user_input)

            with st.chat_message("assistant"):
                st.write(answer_text)
                st.markdown("**Biomedical Evidence (PubMed):**")
                for pmid in sources:
                    st.write(f"• PubMed ID: {pmid}")


            db.cursor.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (st.session_state.active_chat_id, "assistant", answer_text)
            )
            db.conn.commit()

            st.rerun()


st.sidebar.title("BioMed Assistant")

if st.sidebar.button("🧠 Ask Any Medical Question"):
    # Create a new empty chat
    db.cursor.execute("INSERT INTO chats (title, user_id) VALUES ('New Chat', ?)", (st.session_state.user_id,))
    db.conn.commit()

    # Set this chat as active
    st.session_state.active_chat_id = db.cursor.lastrowid

    # Go to chat page
    st.session_state.page = "chat"
    st.rerun()

if st.sidebar.button("📊 Predict Diabetes Risk"):
    st.session_state.page = "risk"
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Previous Conversations")

# if st.sidebar.button("+ New Chat"):
#     db.cursor.execute("INSERT INTO chats (title) VALUES ('New Chat')")
#     db.conn.commit()
#     st.session_state.active_chat_id = db.cursor.lastrowid
#     st.session_state.page = "chat"
#     st.rerun()

db.cursor.execute(
    "SELECT chat_id, title FROM chats WHERE user_id=? ORDER BY created_at DESC",(st.session_state.user_id,)
)
chats = db.cursor.fetchall()

for chat_id, title in chats:
    if st.sidebar.button(title, key=f"chat_{chat_id}"):
        st.session_state.active_chat_id = chat_id
        st.session_state.page = "chat"
        st.rerun()



# -------- Risk Prediction Section --------
# st.divider()
if st.session_state.page == "risk":

    st.title("Diabetes Risk Prediction")

    age = st.number_input("Age", 1, 100)
    bmi = st.number_input("BMI", 10.0, 50.0)
    high_bp = st.selectbox("High Blood Pressure", [0, 1])
    high_chol = st.selectbox("High Cholesterol", [0, 1])
    smoker = st.selectbox("Smoker", [0, 1])

    features = np.zeros(21)
    features[0] = high_bp
    features[1] = high_chol
    features[2] = bmi
    features[3] = smoker
    features[8] = age

    if st.button("Predict Risk"):
        prob = risk_model.predict_proba([features])[0][1]

        if prob < 0.33:
            risk = "Low"
            st.success("Low Risk")
        elif prob < 0.66:
            risk = "Medium"
            st.warning("Medium Risk")
        else:
            risk = "High"
            st.error("High Risk")

        st.write(f"Confidence Score: {prob:.2f}")

        st.subheader("Key Contributing Factors")
        for f in explain_risk(features):
            st.write(f"• {f}")

        st.subheader("Personalized Health Guidance")
        st.write(risk_guidance(risk))
