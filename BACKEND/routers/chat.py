from fastapi import APIRouter, HTTPException
from schema.common import ChatRequest, ChatResponse
from services.llm import ask_general_question
from services.emergency import detect_emergency
from services.safety import build_chat_response

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Handles general health questions.
    Validates against emergency keywords and safety rules.
    """
    message = payload.message.strip()
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    # Step 1: Emergency check
    if detect_emergency(message):
        response_dict = build_chat_response(reply="", is_emergency=True, source="general")
        return ChatResponse(**response_dict)
        
    # Step 2: Query the LLM for general health info
    raw_reply = ask_general_question(message)
    
    # Step 3: Pass reply through safety filter and attach general disclaimer
    response_dict = build_chat_response(reply=raw_reply, is_emergency=False, source="general")
    
    return ChatResponse(**response_dict)
