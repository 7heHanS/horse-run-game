from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai import find_best_move
import json
import os
from datetime import datetime

app = FastAPI(title="Horse Run Game API")

# Configure CORS for GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, you should restrict this to the exact github pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoveRequest(BaseModel):
    board: list[list[int]]
    current_turn: int
    difficulty: int = 3
    use_ml: bool = False

class MoveResponse(BaseModel):
    from_r: int
    from_c: int
    to_r: int
    to_c: int

class MoveEntry(BaseModel):
    player: int
    from_pos: dict  # {"x": int, "y": int}  — aliased from "from" in frontend
    to: dict        # {"x": int, "y": int}

    class Config:
        # Allow 'from' field name from frontend JSON
        populate_by_name = True

class GameLogRequest(BaseModel):
    winner: int
    total_moves: int
    moves: list[dict]  # raw moves from frontend
    timestamp: str

GAME_LOG_FILE = "game_logs.jsonl"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Horse Run Game AI Server is running"}

@app.post("/api/move", response_model=MoveResponse)
def get_next_move(request: MoveRequest):
    # Determine players (assume AI is current_turn)
    ai_player = request.current_turn
    human_player = 1 if ai_player == 2 else 2
    
    # Calculate best move with depth = difficulty + 1 (arbitrary scaling)
    depth = request.difficulty
    
    use_mcts = request.difficulty >= 6
    mcts_sims = 5000 if request.difficulty == 6 else 0

    best_move = find_best_move(
        request.board, 
        ai_player, 
        human_player, 
        depth, 
        use_ml=request.use_ml,
        use_mcts=use_mcts,
        mcts_simulations=mcts_sims
    )
    
    if best_move is None:
        raise HTTPException(status_code=400, detail="No possible moves found")
        
    return {
        "from_r": best_move["pieceY"],
        "from_c": best_move["pieceX"],
        "to_r": best_move["targetY"],
        "to_c": best_move["targetX"]
    }

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

