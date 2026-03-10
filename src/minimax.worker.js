/**
 * Web Worker for Minimax & MCTS (ONNX) AI computation.
 */
import * as ort from 'onnxruntime-web';
import { MCTS } from './mcts.js';
import { BOARD_SIZE, OASIS_POS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves, checkWinCondition } from './engine.js';
import { evaluateBoardV7 } from './heuristics.v7.js';

// === Constants ===
const MEADOW_POSITIONS = [
    { x: 5, y: 4 }, { x: 5, y: 6 }, { x: 4, y: 5 }, { x: 6, y: 5 },
    { x: 5, y: 3 }, { x: 5, y: 7 }, { x: 3, y: 5 }, { x: 7, y: 5 },
    { x: 4, y: 4 }, { x: 6, y: 4 }, { x: 4, y: 6 }, { x: 6, y: 6 }
];

const WEIGHTS = {
    WIN: 100000,
    MEADOW: 500,
    SETUP_THREAT: 1500,
    BLOCKING: 1000,
    CENTER_CONTROL: 10,
    MOBILITY: 1
};

// Global session for MCTS
let onnxSession = null;

async function initONNX() {
    if (onnxSession) return onnxSession;
    try {
        // Path relative to the public root where Vite copies wasm files
        // In local dev, base is /horse-run-game/
        const modelUrl = './model_v6.onnx';
        onnxSession = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'], // Use WebAssembly for worker reliability
            graphOptimizationLevel: 'all'
        });
        console.log("ONNX Session created in worker");
        return onnxSession;
    } catch (e) {
        console.error("Failed to load ONNX model in worker:", e);
        return null;
    }
}

// === Legacy Minimax Logic (Duplicate for isolation) ===

function simulateMove(board, move) {
    const nextBoard = board.map(row => [...row]);
    const player = nextBoard[move.pieceY][move.pieceX];
    nextBoard[move.pieceY][move.pieceX] = 0;
    nextBoard[move.targetY][move.targetX] = player;
    return nextBoard;
}

function getAllPossibleMoves(board, player) {
    const moves = [];
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            if (board[y][x] === player) {
                const slideMoves = getValidSlideMoves(board, x, y);
                const lShapeMoves = getValidLShapeMoves(board, x, y);
                for (const m of slideMoves.concat(lShapeMoves)) {
                    moves.push({
                        pieceX: x, pieceY: y,
                        targetX: m.x, targetY: m.y,
                        type: m.type, player: player
                    });
                }
            }
        }
    }
    moves.sort((a, b) => {
        const da = Math.abs(a.targetX - 5) + Math.abs(a.targetY - 5);
        const db = Math.abs(b.targetX - 5) + Math.abs(b.targetY - 5);
        return da - db;
    });
    return moves;
}

function minimax(board, depth, alpha, beta, isMaximizing, aiPlayer, humanPlayer) {
    if (depth === 0) return evaluateBoardV7(board, aiPlayer, humanPlayer);

    const currentPlayer = isMaximizing ? aiPlayer : humanPlayer;
    const possibleMoves = getAllPossibleMoves(board, currentPlayer);

    if (possibleMoves.length === 0) return evaluateBoardV7(board, aiPlayer, humanPlayer);

    if (isMaximizing) {
        let maxEval = -Infinity;
        for (const move of possibleMoves) {
            const nextBoard = simulateMove(board, move);
            if (checkWinCondition(move.targetX, move.targetY)) return WEIGHTS.WIN + depth;
            const evalScore = minimax(nextBoard, depth - 1, alpha, beta, false, aiPlayer, humanPlayer);
            maxEval = Math.max(maxEval, evalScore);
            alpha = Math.max(alpha, evalScore);
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const move of possibleMoves) {
            const nextBoard = simulateMove(board, move);
            if (checkWinCondition(move.targetX, move.targetY)) return -WEIGHTS.WIN - depth;
            const evalScore = minimax(nextBoard, depth - 1, alpha, beta, true, aiPlayer, humanPlayer);
            minEval = Math.min(minEval, evalScore);
            beta = Math.min(beta, evalScore);
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

function findBestMoveMinimax(board, aiPlayer, humanPlayer, depth) {
    let bestScore = -Infinity;
    let bestMove = null;
    const possibleMoves = getAllPossibleMoves(board, aiPlayer);
    for (const move of possibleMoves) {
        const nextBoard = simulateMove(board, move);
        if (checkWinCondition(move.targetX, move.targetY)) return move;
        const score = minimax(nextBoard, depth - 1, -Infinity, Infinity, false, aiPlayer, humanPlayer);
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}

// === Worker message handler ===
self.onmessage = async function(e) {
    const { board, aiPlayer, humanPlayer, depth, useMCTS } = e.data;

    if (useMCTS) {
        const session = await initONNX();
        if (session) {
            // MCTS logic
            const sims = depth >= 6 ? 2000 : 800; // Scalable simulations
            const mcts = new MCTS(session, ort, 2.5, sims);
            const bestMove = await mcts.search(board, aiPlayer);
            self.postMessage({ bestMove });
            return;
        } else {
            console.warn("ONNX failed, falling back to Minimax");
        }
    }

    // Default to Minimax
    const bestMove = findBestMoveMinimax(board, aiPlayer, humanPlayer, depth);
    self.postMessage({ bestMove });
};
