import { BOARD_SIZE, OASIS_POS, MEADOW_POSITIONS } from './constants.js';

/**
 * 특정 좌표가 '사막(Desert)'인지 판별
 * 사막 = 보드 내에서 오아시스와 초원을 제외한 모든 칸
 */
export function isDesertSpace(x, y) {
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return false;
    if (x === OASIS_POS.x && y === OASIS_POS.y) return false;

    for (const m of MEADOW_POSITIONS) {
        if (m.x === x && m.y === y) return false;
    }

    return true;
}

/**
 * 특정 위치에서 유효한 슬라이드 이동 경로 계산
 * - 벽이나 다른 말에 닿기 직전까지 직진
 * - 최소 1칸 이상 이동 가능한 경우에만 결과 포함
 */
export function getValidSlideMoves(board, startX, startY) {
    const moves = [];
    const directions = [
        { dx: 0, dy: -1 }, // up
        { dx: 0, dy: 1 },  // down
        { dx: -1, dy: 0 }, // left
        { dx: 1, dy: 0 }   // right
    ];

    for (const dir of directions) {
        let x = startX;
        let y = startY;
        let lastValidX = startX;
        let lastValidY = startY;

        while (true) {
            x += dir.dx;
            y += dir.dy;

            // 보드 경계 체크
            if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) {
                break;
            }

            // 장애물 (다른 말) 체크
            if (board[y][x] !== null) {
                break;
            }

            lastValidX = x;
            lastValidY = y;
        }

        // 제자리가 아니라면 이동 가능
        if (lastValidX !== startX || lastValidY !== startY) {
            moves.push({ x: lastValidX, y: lastValidY, type: 'slide' });
        }
    }

    return moves;
}

/**
 * 특정 위치에서 유효한 L자 이동 경로 계산
 * - 체스 나이트(Knight) 이동 방식
 * - 도착지가 비어있는 '사막'이어야 함
 */
export function getValidLShapeMoves(board, startX, startY) {
    const moves = [];
    const knightMoves = [
        { dx: -1, dy: -2 }, { dx: 1, dy: -2 },
        { dx: -2, dy: -1 }, { dx: 2, dy: -1 },
        { dx: -2, dy: 1 }, { dx: 2, dy: 1 },
        { dx: -1, dy: 2 }, { dx: 1, dy: 2 }
    ];

    for (const move of knightMoves) {
        const x = startX + move.dx;
        const y = startY + move.dy;

        // 보드 경계 체크
        if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) continue;

        // 도착지가 비어있는지 체크
        if (board[y][x] !== null) continue;

        // 도착지가 '사막'이어야 함
        if (isDesertSpace(x, y)) {
            moves.push({ x, y, type: 'lshape' });
        }
    }

    return moves;
}

/**
 * 승리 여부 판단
 * - 슬라이드 이동 후 현재 좌표가 오아시스인 경우 리턴 (L자 이동은 애초에 오아시스 진입 불가)
 */
export function checkWinCondition(x, y) {
    return x === OASIS_POS.x && y === OASIS_POS.y;
}
