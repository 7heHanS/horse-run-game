import { BOARD_SIZE, MEADOW_POSITIONS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves } from './engine.js';

const WEIGHTS = {
    "WIN": 100000,
    "MEADOW": 500,
    "SETUP_THREAT": 1500,
    "BLOCKING": 1000,
    "CENTER_CONTROL": 10,
    "MOBILITY": 1
};

/**
 * Port of V7 evaluate_board from ai.py
 */
export function evaluateBoardV7(board, aiPlayer, humanPlayer) {
    let score = 0.0;
    
    // 1. Coordinate-based scores
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            const p = board[y][x];
            if (p === 0 || p === null) continue;
            
            const dist = Math.abs(x - 5) + Math.abs(y - 5);
            // Base center control score
            const centerScore = (10 - dist) * WEIGHTS["CENTER_CONTROL"];
            
            // Proximity threat: extra weights for being very close to winning
            let threatScore = 0;
            if (dist <= 8) {
                threatScore = (10 - dist) * 2000;
            }

            if (p === aiPlayer) {
                score += (centerScore + threatScore);
            } else if (p === humanPlayer) {
                score -= (centerScore + threatScore);
            }
        }
    }

    // 2. Axis Patrol & Stopper Detection
    const stoppers = [
        { sx: 6, sy: 5, side: "L" }, { sx: 4, sy: 5, side: "R" },
        { sx: 5, sy: 6, side: "T" }, { sx: 5, sy: 4, side: "B" }
    ];
    
    let aiOnX5 = 0, humanOnX5 = 0;
    let aiOnY5 = 0, humanOnY5 = 0;
    
    for (let i = 0; i < BOARD_SIZE; i++) {
        // board[y][x]
        const rowVal = board[5][i];
        if (rowVal === aiPlayer) aiOnX5++;
        else if (rowVal === humanPlayer) humanOnX5++;
        
        const colVal = board[i][5];
        if (colVal === aiPlayer) aiOnY5++;
        else if (colVal === humanPlayer) humanOnY5++;
    }

    // Axis Patrol Reward
    score += (aiOnX5 + aiOnY5) * 2000;
    score -= (humanOnX5 + humanOnY5) * 3000;

    // Cluster Penalty
    if (humanOnX5 > 1) score -= 10000;
    if (humanOnY5 > 1) score -= 10000;

    for (const { sx, sy, side } of stoppers) {
        const pAtStopper = board[sy][sx];
        
        for (const player of [1, 2]) {
            let isImminent = false;
            let isBlockedThreat = false;
            
            if (side === "L") {
                for (let x = 0; x < 5; x++) {
                    if (board[5][x] === player) {
                        let clear = true;
                        for (let k = x + 1; k < 5; k++) {
                            if (board[5][k] !== 0 && board[5][k] !== null) { clear = false; break; }
                        }
                        if (clear) { isImminent = true; break; }
                        else isBlockedThreat = true;
                    }
                }
            } else if (side === "R") {
                for (let x = 6; x < 11; x++) {
                    if (board[5][x] === player) {
                        let clear = true;
                        for (let k = 5 + 1; k < x; k++) {
                            if (board[5][k] !== 0 && board[5][k] !== null) { clear = false; break; }
                        }
                        if (clear) { isImminent = true; break; }
                        else isBlockedThreat = true;
                    }
                }
            } else if (side === "T") {
                for (let y = 0; y < 5; y++) {
                    if (board[y][5] === player) {
                        let clear = true;
                        for (let k = y + 1; k < 5; k++) {
                            if (board[k][5] !== 0 && board[k][5] !== null) { clear = false; break; }
                        }
                        if (clear) { isImminent = true; break; }
                        else isBlockedThreat = true;
                    }
                }
            } else if (side === "B") {
                for (let y = 6; y < 11; y++) {
                    if (board[y][5] === player) {
                        let clear = true;
                        for (let k = 5 + 1; k < y; k++) {
                            if (board[k][5] !== 0 && board[k][5] !== null) { clear = false; break; }
                        }
                        if (clear) { isImminent = true; break; }
                        else isBlockedThreat = true;
                    }
                }
            }
            
            let threatLevel = 0;
            if (isImminent && pAtStopper !== 0 && pAtStopper !== null) threatLevel = 50000;
            else if (isImminent && (pAtStopper === 0 || pAtStopper === null)) threatLevel = 15000;
            else if (isBlockedThreat && pAtStopper !== 0 && pAtStopper !== null) threatLevel = 8000;
            
            if (player === aiPlayer) score += threatLevel;
            else score -= threatLevel;
        }
    }

    // 3. Mobility
    const aiMoves = getAllPossibleMovesCount(board, aiPlayer);
    const humanMoves = getAllPossibleMovesCount(board, humanPlayer);
    score += (aiMoves - humanMoves) * WEIGHTS["MOBILITY"];
    
    return score;
}

function getAllPossibleMovesCount(board, player) {
    let count = 0;
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            const p = board[y][x];
            if (p === player) {
                count += getValidSlideMoves(board, x, y).length;
                count += getValidLShapeMoves(board, x, y).length;
            }
        }
    }
    return count;
}
