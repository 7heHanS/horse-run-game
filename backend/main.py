from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from datetime import datetime

app = FastAPI(title="Horse Run Game Log Server")

# Configure CORS for GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, you should restrict this to the exact github pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameLogRequest(BaseModel):
    winner: int
    total_moves: int
    moves: list[dict]  # raw moves from frontend
    timestamp: str

GAME_LOG_FILE = "game_logs.jsonl"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Horse Run Game Log Server is running"}

@app.post("/api/game-log")
def receive_game_log(request: GameLogRequest):
    """Receive and store a winning game record for future model training."""
    log_entry = {
        "winner": request.winner,
        "total_moves": request.total_moves,
        "moves": request.moves,
        "client_timestamp": request.timestamp,
        "server_timestamp": datetime.now().isoformat(),
    }
    
    with open(GAME_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    print(f"[GAME LOG] Winner: P{request.winner}, Moves: {request.total_moves}, Time: {request.timestamp}")
    return {"status": "ok", "message": "Game log recorded"}
