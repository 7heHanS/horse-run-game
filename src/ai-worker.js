// ai-worker.js

// 간단한 상태 복사를 위해 엔진 함수를 재정의하거나 임포트해야 합니다.
// Web Worker 내에서는 DOM이나 main script의 모듈을 쉽게 import 할 수 있지만,
// Vite 환경에서는 type: 'module' worker로 불러올 수 있습니다.

import { BOARD_SIZE, OASIS_POS, MEADOW_POSITIONS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves, checkWinCondition } from './engine.js';

// 평가 함수 가중치
const WEIGHTS = {
    WIN: 10000,
    MEADOW: 100,
    MOBILITY: 1
};

self.onmessage = function (e) {
    const { boardInfo, pieces, currentPlayer, depth } = e.data;

    // AI는 항상 2라고 가정
    const aiPlayer = 2;
    const humanPlayer = 1;

    console.log(`[AI Worker] Calculating best move at depth ${depth}`);
    const bestMove = findBestMove(boardInfo, pieces, aiPlayer, humanPlayer, depth);

    self.postMessage({ type: 'MOVE_CALCULATED', move: bestMove });
};

function findBestMove(board, pieces, aiPlayer, humanPlayer, depth) {
    let bestScore = -Infinity;
    let bestMove = null;

    const possibleMoves = getAllPossibleMoves(board, aiPlayer);

    // 무작위성을 약간 섞어 매번 똑같은 플레이를 하지 않도록 함
    possibleMoves.sort(() => Math.random() - 0.5);

    for (const move of possibleMoves) {
        // 보드 상태 복제 및 시뮬레이션
        const { nextBoard, nextPieces } = simulateMove(board, pieces, move);

        // 승리 조건 체크 (조기 종료)
        if (checkWinCondition(move.targetX, move.targetY)) {
            return move;
        }

        const score = minimax(nextBoard, nextPieces, depth - 1, -Infinity, Infinity, false, aiPlayer, humanPlayer);

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }

    return bestMove;
}

function minimax(board, pieces, depth, alpha, beta, isMaximizing, aiPlayer, humanPlayer) {
    if (depth === 0) {
        return evaluateBoard(board, pieces, aiPlayer, humanPlayer);
    }

    const currentPlayer = isMaximizing ? aiPlayer : humanPlayer;
    const possibleMoves = getAllPossibleMoves(board, currentPlayer);

    if (possibleMoves.length === 0) {
        return evaluateBoard(board, pieces, aiPlayer, humanPlayer);
    }

    if (isMaximizing) {
        let maxEval = -Infinity;
        for (const move of possibleMoves) {
            const { nextBoard, nextPieces } = simulateMove(board, pieces, move);

            if (checkWinCondition(move.targetX, move.targetY)) {
                return WEIGHTS.WIN + depth; // 빠른 승리 선호
            }

            const evalScore = minimax(nextBoard, nextPieces, depth - 1, alpha, beta, false, aiPlayer, humanPlayer);
            maxEval = Math.max(maxEval, evalScore);
            alpha = Math.max(alpha, evalScore);
            if (beta <= alpha) break; // 가지치기
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const move of possibleMoves) {
            const { nextBoard, nextPieces } = simulateMove(board, pieces, move);

            if (checkWinCondition(move.targetX, move.targetY)) {
                return -WEIGHTS.WIN - depth; // 플레이어의 빠른 승리 회피
            }

            const evalScore = minimax(nextBoard, nextPieces, depth - 1, alpha, beta, true, aiPlayer, humanPlayer);
            minEval = Math.min(minEval, evalScore);
            beta = Math.min(beta, evalScore);
            if (beta <= alpha) break; // 가지치기
        }
        return minEval;
    }
}

function getAllPossibleMoves(board, player) {
    const moves = [];
    for (let y = 0; y < BOARD_SIZE; y++) {
        for (let x = 0; x < BOARD_SIZE; x++) {
            const piece = board[y][x];
            if (piece && piece.player === player) {
                const slideMoves = getValidSlideMoves(board, x, y);
                const lShapeMoves = getValidLShapeMoves(board, x, y);

                for (const m of [...slideMoves, ...lShapeMoves]) {
                    moves.push({
                        pieceX: x, pieceY: y,
                        targetX: m.x, targetY: m.y,
                        type: m.type, player
                    });
                }
            }
        }
    }
    return moves;
}

function simulateMove(board, pieces, move) {
    // 깊은 복사를 위해 간단히 JSON 사용 (성능 중요시 최적화 필요)
    const nextBoard = board.map(row => [...row]);
    const nextPieces = JSON.parse(JSON.stringify(pieces));

    const piece = nextBoard[move.pieceY][move.pieceX];

    // 이전 위치 비우기
    nextBoard[move.pieceY][move.pieceX] = null;

    // 새 위치 적용
    const movedPiece = nextPieces.find(p => p.x === move.pieceX && p.y === move.pieceY);
    if (movedPiece) {
        movedPiece.x = move.targetX;
        movedPiece.y = move.targetY;
        nextBoard[move.targetY][move.targetX] = movedPiece;
    }

    return { nextBoard, nextPieces };
}

function evaluateBoard(board, pieces, aiPlayer, humanPlayer) {
    let score = 0;

    // 1. 초원(Meadow) 점유 점수
    for (const m of MEADOW_POSITIONS) {
        const piece = board[m.y][m.x];
        if (piece) {
            if (piece.player === aiPlayer) score += WEIGHTS.MEADOW;
            else if (piece.player === humanPlayer) score -= WEIGHTS.MEADOW;
        }
    }

    // 2. 이동 가능 횟수 점수 (Mobility)
    // 연산량이 늘어나므로 깊이가 깊어지면 생략하거나 최적화 가능
    const aiMoves = getAllPossibleMoves(board, aiPlayer).length;
    const humanMoves = getAllPossibleMoves(board, humanPlayer).length;
    score += (aiMoves - humanMoves) * WEIGHTS.MOBILITY;

    return score;
}
