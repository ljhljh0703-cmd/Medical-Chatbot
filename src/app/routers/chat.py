from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/message", response_model=ChatResponse)
def send_message(request: ChatRequest):
    return {"message": "placeholder"}
