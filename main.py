# backend/main.py
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq # Groq Python SDK

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="WellnessHub Backend",
    description="AI-Powered Medical & Emotional Support Assistant Backend",
    version="1.0.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
# This is crucial for allowing your frontend (running on a different origin/port)
# to communicate with your backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. In production, specify your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Serve Static Frontend Files ---
# Define the path to your frontend directory
FRONTEND_DIR = Path("../frontend")

# Mount the static files directory.
# This tells FastAPI to serve files from the 'frontend' directory
# when requests come to the root path ("/").
# html=True ensures that if a directory is requested (like "/"),
# it will automatically serve "index.html" from that directory.
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


# --- Groq API Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Ensure the API key is set in the environment variables.


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# --- API Endpoint for AI Chat ---
@app.post("/api/chat")
async def chat_with_ai(request: Request):
    """
    Receives a user message, sends it to the Groq API, and returns the AI's response.
    """
    try:
        data = await request.json()
        user_message = data.get("message")

        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty."})

        # Define the system prompt for the AI
        # This guides the AI's persona and purpose.
        system_prompt = (
            "You are WellnessHub, an AI-powered medical and emotional support assistant. "
            "Provide helpful, empathetic, and informative responses. "
            "For medical advice, always recommend consulting a qualified healthcare professional. "
            "For emotional support, focus on active listening, validation, and suggesting healthy coping mechanisms or professional help when appropriate. "
            "Keep responses concise and easy to understand."
        )

        # Call the Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model="llama3-8b-8192",  # Or other suitable Groq models like "mixtral-8x7b-32768"
            temperature=0.7, # Controls randomness. Lower for more deterministic, higher for more creative.
            max_tokens=500, # Maximum tokens in the response
        )

        ai_response = chat_completion.choices[0].message.content
        return JSONResponse(content={"response": ai_response})

    except Exception as e:
        print(f"Error processing chat request: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})

# --- Root endpoint (optional, as StaticFiles handles "/") ---
# This can be used for a simple API health check or redirect if needed.
@app.get("/api/health")
async def health_check():
    """
    Basic health check endpoint for the backend API.
    """
    return {"status": "ok", "message": "WellnessHub backend is running!"}

