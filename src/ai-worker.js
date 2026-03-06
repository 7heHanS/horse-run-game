// ai-worker.js

// 간단한 상태 복사를 위해 엔진 함수를 재정의하거나 임포트해야 합니다.
// Web Worker 내에서는 DOM이나 main script의 모듈을 쉽게 import 할 수 있지만,
// Vite 환경에서는 type: 'module' worker로 불러올 수 있습니다.

import { BOARD_SIZE, OASIS_POS, MEADOW_POSITIONS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves, checkWinCondition } from './engine.js';

// 평가 함수 가중치 (세분화)
const WEIGHTS = {
    WIN: 100000,
    MEADOW: 500,        // 초원 점유 (스토퍼)
    SETUP_THREAT: 1500, // 승리 직전 셋업 (초원 점유 + 같은 라인에 발사대 대기)
    BLOCKING: 1000,     // 상대방의 셋업 라인에 내 말을 올려두어 견제
    CENTER_CONTROL: 10, // 보드 중앙(오아시스 주변) 접근성 (초반 전개용)
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

    // 1. 중앙 장악력 (Center Control) - 초반 수 연구 피드백 반영
    // 구석에 짱박혀 있지 않고 중앙으로 빨리 전개하도록 유도
    for (const p of pieces) {
        // 오아시스(5,5)와의 맨해튼 거리 계산
        const distToCenter = Math.abs(p.x - 5) + Math.abs(p.y - 5);
        // 거리가 가까울수록 가점 (최대 거리 10)
        const centerScore = (10 - distToCenter) * WEIGHTS.CENTER_CONTROL;

        if (p.player === aiPlayer) score += centerScore;
        else if (p.player === humanPlayer) score -= centerScore;
    }

    // 2. 초원(Meadow) 점유 및 셋업/견제(Block) 평가
    for (const m of MEADOW_POSITIONS) {
        const stopperPiece = board[m.y][m.x];

        if (stopperPiece) {
            const owner = stopperPiece.player;
            const opponent = (owner === aiPlayer) ? humanPlayer : aiPlayer;
            const sign = (owner === aiPlayer) ? 1 : -1;

            // 초원 기본 점수
            score += WEIGHTS.MEADOW * sign;

            // 3. 발사대(Launcher) 및 견제(Block) 확인
            // 초원이 오아시스의 가로축인지 세로축인지 판단
            const isHorizontal = (m.y === 5); // e6(4,5), f7(6,5) 인 경우 y축이 같으므로 x축(가로) 라인 검사
            const isVertical = (m.x === 5);   // f5(5,4), g6(5,6) 인 경우 x축이 같으므로 y축(세로) 라인 검사

            let ownerLaunchers = 0;
            let opponentBlockers = 0;

            if (isHorizontal) {
                // 같은 행(가로)에 있는 말들 검사
                for (let x = 0; x < BOARD_SIZE; x++) {
                    if (x === m.x || x === 5) continue; // 스토퍼 자신과 오아시스 칸 제외
                    const p = board[m.y][x];
                    if (p) {
                        if (p.player === owner) ownerLaunchers++;
                        if (p.player === opponent) opponentBlockers++;
                    }
                }
            } else if (isVertical) {
                // 같은 열(세로)에 있는 말들 검사
                for (let y = 0; y < BOARD_SIZE; y++) {
                    if (y === m.y || y === 5) continue;
                    const p = board[y][m.x];
                    if (p) {
                        if (p.player === owner) ownerLaunchers++;
                        if (p.player === opponent) opponentBlockers++;
                    }
                }
            }

            // 점수 정산
            if (ownerLaunchers > 0) {
                if (opponentBlockers === 0) {
                    // 방해물 없이 완벽한 셋업이 된 상태 (엄청난 위협)
                    score += WEIGHTS.SETUP_THREAT * sign;
                } else {
                    // 셋업을 시도했으나 상대방이 견제(Block) 중인 상태
                    // 방어자에게 블로킹 점수 부여 (sign을 반대로 적용)
                    score += WEIGHTS.BLOCKING * (-sign);
                }
            }
        }
    }

    // 4. 기동성 (Mobility)
    const aiMoves = getAllPossibleMoves(board, aiPlayer).length;
    const humanMoves = getAllPossibleMoves(board, humanPlayer).length;
    score += (aiMoves - humanMoves) * WEIGHTS.MOBILITY;

    return score;
}
