from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.services import (
    handle_consent,
    get_history,
    delete_history_item,
    delete_all_history,
    update_rating,
    update_feedback,
    save_interaction
)
from app.chatbot import generate_chat_response

app = FastAPI()

# Mount static files (UI)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# CORS (if frontend hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse("frontend/index.html")

@app.post("/consent")
async def consent_endpoint(request: Request):
    data = await request.json()
    return handle_consent(data)

@app.get("/history")
async def history_endpoint(user_id: str):
    return get_history(user_id)

@app.delete("/delete_history_item")
async def delete_item_endpoint(user_id: str, index: int):
    return delete_history_item(user_id, index)

@app.delete("/delete_all_history")
async def delete_all_endpoint(user_id: str):
    return delete_all_history(user_id)

@app.post("/update_rating")
async def rating_endpoint(user_id: str, message_id: int, rating: int):
    return update_rating(user_id, message_id, rating)

@app.post("/update_feedback")
async def feedback_endpoint(user_id: str, message_id: int, feedback: str):
    return update_feedback(user_id, message_id, feedback)

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    response = await generate_chat_response(data)
    
    # Save the interaction to history
    user_id = data.get("user_id", "default_user")
    query = data.get("query", "")
    save_interaction(user_id, query, response)
    
    return response