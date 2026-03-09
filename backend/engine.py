BOARD_SIZE = 11
OASIS_POS = {"x": 5, "y": 5}
MEADOW_POSITIONS = [
    {"x": 5, "y": 4},
    {"x": 5, "y": 6},
    {"x": 4, "y": 5},
    {"x": 6, "y": 5},
    {"x": 5, "y": 3}, {"x": 5, "y": 7}, {"x": 3, "y": 5}, {"x": 7, "y": 5},
    {"x": 4, "y": 4}, {"x": 6, "y": 4}, {"x": 4, "y": 6}, {"x": 6, "y": 6}
]

def is_desert_space(x: int, y: int) -> bool:
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return False
    if x == OASIS_POS["x"] and y == OASIS_POS["y"]:
        return False
    
    for m in MEADOW_POSITIONS:
        if m["x"] == x and m["y"] == y:
            return False
            
    return True

def get_valid_slide_moves(board: list[list[int]], start_x: int, start_y: int):
    # board values: 0 for empty, 1 for player 1, 2 for player 2
    moves = []
    directions = [
        {"dx": 0, "dy": -1}, # up
        {"dx": 0, "dy": 1},  # down
        {"dx": -1, "dy": 0}, # left
        {"dx": 1, "dy": 0}   # right
    ]
    
    for d in directions:
        x, y = start_x, start_y
        last_valid_x, last_valid_y = start_x, start_y
        
        while True:
            x += d["dx"]
            y += d["dy"]
            
            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                break
                
            if board[y][x] != 0:
                break
                
            last_valid_x, last_valid_y = x, y
            
        if last_valid_x != start_x or last_valid_y != start_y:
            moves.append({"x": last_valid_x, "y": last_valid_y, "type": "slide"})
            
    return moves

def get_valid_l_shape_moves(board: list[list[int]], start_x: int, start_y: int):
    moves = []
    knight_moves = [
        {"dx": -1, "dy": -2}, {"dx": 1, "dy": -2},
        {"dx": -2, "dy": -1}, {"dx": 2, "dy": -1},
        {"dx": -2, "dy": 1}, {"dx": 2, "dy": 1},
        {"dx": -1, "dy": 2}, {"dx": 1, "dy": 2}   
    ]
    
    for m in knight_moves:
        x = start_x + m["dx"]
        y = start_y + m["dy"]
        
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            continue
            
        if board[y][x] != 0:
            continue
            
        if is_desert_space(x, y):
            moves.append({"x": x, "y": y, "type": "lshape"})
            
    return moves

def check_win_condition(x: int, y: int) -> bool:
    return x == OASIS_POS["x"] and y == OASIS_POS["y"]
