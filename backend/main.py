from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai import find_best_move

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
    
    best_move = find_best_move(
        request.board, 
        ai_player, 
        human_player, 
        depth, 
        use_ml=request.use_ml,
        use_mcts=request.use_ml,
        mcts_simulations=3000
    )
    
    if best_move is None:
        raise HTTPException(status_code=400, detail="No possible moves found")
        
    return {
        "from_r": best_move["pieceY"],
        "from_c": best_move["pieceX"],
        "to_r": best_move["targetY"],
        "to_c": best_move["targetX"]
    }
