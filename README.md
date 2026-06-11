Biomedical Knowledge and Risk Analysis System

An AI-powered healthcare assistant that combines Retrieval-Augmented Generation (RAG), Biomedical NLP, and Machine Learning to provide evidence-based medical insights and personalized health risk analysis.

🚀 Features
🔍 Biomedical Question Answering
Retrieves relevant biomedical literature using semantic search.
Generates context-aware responses grounded in retrieved evidence.
🧬 Domain-Specific Retrieval
Uses PubMedBERT embeddings for biomedical semantic understanding.
Stores embeddings in FAISS for efficient vector similarity search.
💬 ChatGPT-like Conversational Interface
Supports multi-turn conversations and follow-up questions.
Maintains contextual memory across a chat session.
Persistent chat history with conversation-based navigation.
👥 Multi-User Support
User registration and login system.
Isolated chat histories for each user using SQLite.
📊 Diabetes Risk Prediction
Machine Learning-based risk assessment using health indicators.
Provides confidence scores, key contributing factors, and personalized health guidance.
🛡️ AI Safety Layer
Detects self-harm and suicide-related queries.
Prevents unsafe responses and provides supportive crisis guidance.
Logs safety alerts for administrative review.


🏗️ System Architecture
User Query
     │
     ▼
PubMedBERT Embedding
     │
     ▼
FAISS Vector Search
     │
     ▼
Relevant Biomedical Documents
     │
     ▼
Gemini LLM
     │
     ▼
Evidence-Based Response


Risk Analysis Pipeline
User Health Data
        │
        ▼
Random Forest Model
        │
        ▼
Risk Prediction
        │
        ▼
AI-Generated Health Guidance


🛠️ Tech Stack
Frontend: Streamlit
LLM: Gemini Flash
Embeddings: PubMedBERT
Vector Database: FAISS
Machine Learning: Scikit-Learn (Random Forest)
Database: SQLite
Language: Pytho


📂 Project Components
RAG Module
Semantic retrieval using PubMedBERT embeddings.
Evidence-based biomedical response generation.
Risk Prediction Module
Diabetes risk classification.
Explainable predictions using feature importance.
Conversation Management
Multi-user authentication.
Persistent chat sessions.
Context-aware follow-up interactions.
Safety Monitoring
Harmful query detection.
Ethical response generation.
Administrative alert logging.

🎯 Key Highlights
Biomedical domain-specific retrieval using PubMedBERT.
Evidence-grounded AI responses instead of generic LLM outputs.
Integration of Machine Learning and Generative AI.
Explainable and safety-aware healthcare assistant.
ChatGPT-inspired user experience with persistent conversation memory.
🔮 Future Enhancements
Support for additional disease risk prediction models.
Hybrid retrieval (keyword + vector search).
Real-time PubMed integration.
Evidence confidence scoring.
Clinical decision support dashboard.
Email/SMS notifications for critical safety alerts.
📌 Disclaimer

This project is intended for educational and research purposes only. It does not provide medical diagnosis, treatment, or professional healthcare advice. Users should consult qualified healthcare professionals for medical decisions.
