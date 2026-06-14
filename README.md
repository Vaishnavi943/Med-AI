# AI Medical Information Chatbot Backend

This is the backend repository for the AI Medical Information Chatbot. It is built using **FastAPI** and **Google Gemini**, providing a safe, strictly controlled conversational interface for general health queries, as well as a local RAG (Retrieval-Augmented Generation) pipeline for querying uploaded medical PDFs.

> **Disclaimer**: This is an educational project. The chatbot does not provide medical diagnoses, prescriptions, or treatment decisions.

---

## 🏗️ Architecture & Responsibilities

The backend was divided into three core responsibilities:

1. **Person A (General Chat)**: Handles the `/api/chat` endpoint. Integrates with the Gemini LLM to answer basic health, symptom, and medicine queries safely.
2. **Person B (PDF & RAG)**: Handles the `/api/upload-pdf` and `/api/ask-pdf` endpoints. Extracts text from PDFs using `PyMuPDF`, chunks and embeds the text using `sentence-transformers`, and answers questions based *only* on the document context.
3. **Person C (Safety & Middleware)**: The core gatekeeper. Checks every incoming message for emergency keywords, filters every outgoing response for unsafe language (like diagnosis or dosage instructions), attaches required medical disclaimers, and handles API rate limiting.

---

## 📂 Project Structure

```
BACKEND/
│
├── main.py                  # FastAPI entry point. Mounts all routers and middleware.
├── requirements.txt         # Project dependencies.
├── .env                     # Environment variables (API keys, settings).
│
├── config/
│   └── settings.py          # Central configuration (Prompts, Disclaimers, Emergency Keywords).
│
├── middleware/
│   ├── logging.py           # Request/response logging middleware.
│   └── rate_limits.py       # API abuse prevention.
│
├── routers/
│   ├── chat.py              # Endpoint for general health questions.
│   └── pdf.py               # Endpoints for uploading PDFs and asking PDF-specific questions.
│
├── schema/
│   └── common.py            # Pydantic models (Data contracts for Requests/Responses).
│
└── services/
    ├── emergency.py         # Detects emergency medical situations based on keywords.
    ├── safety.py            # Filters unsafe LLM outputs and attaches disclaimers.
    ├── llm.py               # Handles the Gemini API calls for general chat.
    ├── pdf_parser.py        # Extracts text from uploaded PDFs via PyMuPDF.
    ├── embeddings.py        # Generates vector embeddings via sentence-transformers locally.
    └── rag.py               # In-memory vector store and retrieval logic for PDFs.
```

---

## 🚀 How to Run Locally

### 1. Prerequisites
You need Python 3.9+ installed on your machine.

### 2. Setup the Environment
Navigate to the `BACKEND` directory in your terminal and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Add your API Key
Open the `.env` file inside the `BACKEND` folder and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Start the Server
Run the FastAPI development server using uvicorn:
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

---

## 🧪 Testing the APIs

You do not need the frontend to test the backend. FastAPI provides a built-in Swagger UI where you can test all endpoints interactively.

1. Open your browser and go to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.
2. You will see three main endpoints:

### Endpoint 1: `POST /api/chat`
* **Purpose**: Ask general health questions.
* **How to test**: Click "Try it out", enter a message like `"What are the symptoms of dengue?"` or `"I have chest pain"`, and hit Execute. Notice how the safety filter and disclaimers are automatically applied.

### Endpoint 2: `POST /api/upload-pdf`
* **Purpose**: Upload a medical report (PDF).
* **How to test**: Click "Try it out", upload a PDF file from your computer, and hit Execute. It will return a `session_id`. **Copy this ID**.

### Endpoint 3: `POST /api/ask-pdf`
* **Purpose**: Ask questions specifically about the document you just uploaded.
* **How to test**: Click "Try it out", paste the `session_id` you copied, type a question like `"What medicines are mentioned in this prescription?"`, and hit Execute.