import { INITIAL_POSITIONS, BOARD_SIZE, OASIS_POS, MEADOW_POSITIONS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves, checkWinCondition } from './engine.js';

class HorseRunGame {
    constructor() {
        this.boardElement = document.getElementById('game-board');
        this.statusBar = document.getElementById('status-bar');
        this.resetBtn = document.getElementById('reset-btn');

        this.boardInfo = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null));
        this.pieces = [];
        this.currentPlayer = 1;
        this.selectedPiece = null;
        this.validMoves = [];
        this.isGameOver = false;
        this.isAITurn = false;

        // Game mode settings
        this.gameMode = '1p';  // '1p' or '2p'
        this.humanPlayer = 1;  // Which player the human controls (1=선공, 2=후공)
        this.aiPlayer = 2;     // Which player the AI controls

        // 복기 기능 상태 변수
        this.moveHistory = [];
        this.isReplayMode = false;
        this.currentReplayStep = 0;

        // 기보 로깅 (개별 수 기록)
        this.gameLog = [];

        // Persistent AI Worker & Preload State
        this.aiWorker = null;
        this.isAIReady = false;

        this.init();
    }

    init() {
        // 복기 UI 바인딩
        this.replayControls = document.getElementById('replay-controls');
        this.prevBtn = document.getElementById('prev-btn');
        this.nextBtn = document.getElementById('next-btn');
        this.exitReplayBtn = document.getElementById('exit-replay-btn');
        this.replayStatus = document.getElementById('replay-status');

        this.prevBtn?.addEventListener('click', () => this.replayPrev());
        this.nextBtn?.addEventListener('click', () => this.replayNext());
        this.exitReplayBtn?.addEventListener('click', () => this.exitReplay());

        this.createBoardUI();
        this.initPieces();
        this.renderPieces();

        this.resetBtn.addEventListener('click', () => {
            this.resetGame();
        });

        // Segmented button click handler (generic for all segment groups)
        document.querySelectorAll('.segment-group').forEach(group => {
            group.querySelectorAll('.seg-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    // Update active state
                    group.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    group.dataset.value = btn.dataset.val;

                    // If game mode changed, toggle visibility of AI-only controls
                    if (group.id === 'game-mode-group') {
                        const is2P = btn.dataset.val === '2p';
                        document.getElementById('turn-order-group').style.display = is2P ? 'none' : '';
                        document.getElementById('ai-mode-group').style.display = is2P ? 'none' : '';
                        document.getElementById('ai-power-group').style.display = is2P ? 'none' : '';
                        this.resetGame();
                    }
                    
                    if (group.id === 'ai-mode-seg') {
                        this.updateSliderRange();
                    }
                });
            });
        });

        const powerSlider = document.getElementById('ai-power-slider');
        const powerValue = document.getElementById('ai-power-value');
        if (powerSlider) {
            powerSlider.addEventListener('input', (e) => {
                powerValue.textContent = e.target.value;
            });
        }

        this.initWorker();
    }

    updateSliderRange() {
        const aiModeGroup = document.getElementById('ai-mode-seg');
        const mode = aiModeGroup?.dataset.value || 'minimax';
        const slider = document.getElementById('ai-power-slider');
        const powerValue = document.getElementById('ai-power-value');
        
        if (!slider) return;

        if (mode === 'minimax') {
            slider.min = '2';
            slider.max = '5';
            slider.value = '3';
            slider.step = '1';
        } else { // mcts
            slider.min = '100';
            slider.max = '2000';
            slider.value = '400';
            slider.step = '100';
        }
        powerValue.textContent = slider.value;
    }

    initWorker() {
        if (!this.aiWorker) {
            this.aiWorker = new Worker(
                new URL('./minimax.worker.js', import.meta.url),
                { type: 'module' }
            );
            
            this.aiWorker.onmessage = (e) => {
                if (e.data.type === 'ready') {
                    console.log("AI Background Model is fully loaded and ready.");
                    this.isAIReady = true;
                    this.hideLoadingOverlay();
                    
                    // If AI was waiting to move on its turn, trigger it now.
                    if (this.isAITurn && this.pendingAITask) {
                        this.pendingAITask();
                        this.pendingAITask = null;
                    }
                } else if (e.data.bestMove !== undefined && this.onAIMoveReceived) {
                    this.onAIMoveReceived(e.data.bestMove);
                    this.onAIMoveReceived = null; // Reset callback
                }
            };
            
            this.aiWorker.onerror = (err) => {
                console.error("Persistent AI Worker error:", err);
            };

            // Trigger eager background preload immediately
            this.aiWorker.postMessage({ type: 'preload' });
        }
    }

    showLoadingOverlay() {
        const overlay = document.getElementById('ai-loading-overlay');
        if (overlay) overlay.classList.remove('hidden');
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('ai-loading-overlay');
        if (overlay) overlay.classList.add('hidden');
    }

    createBoardUI() {
        this.boardElement.innerHTML = '';

        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.x = x;
                cell.dataset.y = y;

                // 타일 유형 설정
                if (x === OASIS_POS.x && y === OASIS_POS.y) {
                    cell.classList.add('oasis');
                } else if (MEADOW_POSITIONS.some(p => p.x === x && p.y === y)) {
                    cell.classList.add('meadow');
                }

                cell.addEventListener('click', () => this.handleCellClick(x, y));

                this.boardElement.appendChild(cell);
            }
        }
    }

    initPieces() {
        this.pieces = [];
        this.boardInfo = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null));

        INITIAL_POSITIONS.PLAYER_1.forEach((pos, i) => {
            this.addPiece(pos.x, pos.y, 1, `p1_${i}`);
        });

        INITIAL_POSITIONS.PLAYER_2.forEach((pos, i) => {
            this.addPiece(pos.x, pos.y, 2, `p2_${i}`);
        });

        // DOM에서 기존 말 초기화
        document.querySelectorAll('.piece').forEach(p => p.remove());

        // 보드에 처음 말 엘리먼트들 마운트 (이후엔 DOM 파괴없이 위치만 교체)
        this.pieces.forEach(piece => {
            const pieceEl = document.createElement('div');
            pieceEl.id = piece.id;
            pieceEl.className = `piece player${piece.player}`;
            this.boardElement.appendChild(pieceEl);
            this.updatePieceView(piece);
        });

        // 초기 상태 저장
        this.moveHistory = [];
        this.saveHistoryState();
    }

    saveHistoryState() {
        // 객체 깊은 복사로 현재 상태를 기록
        this.moveHistory.push(JSON.parse(JSON.stringify(this.pieces)));
    }

    addPiece(x, y, player, id) {
        const piece = { x, y, player, id };
        this.pieces.push(piece);
        this.boardInfo[y][x] = piece;
    }

    updatePieceView(piece) {
        const el = document.getElementById(piece.id);
        if (!el) return;

        const cellPercent = 100 / 11;
        el.style.left = `calc(${piece.x * cellPercent}% + 2px)`;
        el.style.top = `calc(${piece.y * cellPercent}% + 2px)`;

        if (this.selectedPiece === piece) {
            el.classList.add('selected');
        } else {
            el.classList.remove('selected');
        }
    }

    renderPieces() {
        this.pieces.forEach(piece => {
            this.updatePieceView(piece);
        });
    }

    getCellElement(x, y) {
        return document.querySelector(`.cell[data-x="${x}"][data-y="${y}"]`);
    }

    handleCellClick(x, y) {
        if (this.isGameOver || this.isAITurn) return;

        // If user wants to move but AI is not ready, block UX
        const aiModeGroup = document.getElementById('ai-mode-seg');
        const aiMode = aiModeGroup?.dataset.value || 'minimax';
        
        // Only block play if DL mode is selected and models are downloading.
        if (aiMode === 'mcts' && !this.isAIReady) {
            this.showLoadingOverlay();
            return;
        }

        // 선택 가능한 이동 경로 클릭 시
        const move = this.validMoves.find(m => m.x === x && m.y === y);
        if (move) {
            this.movePiece(this.selectedPiece, move.x, move.y);
            return;
        }

        // 현재 플레이어의 말 클릭 시
        const clickedPiece = this.boardInfo[y][x];
        if (clickedPiece && clickedPiece.player === this.currentPlayer) {
            this.selectPiece(clickedPiece);
            return;
        }

        // 빈 공간 클릭이나 유효하지 않은 클릭 시 선택 취소
        this.clearSelection();
    }

    selectPiece(piece) {
        this.selectedPiece = piece;
        this.calculateValidMoves(piece);
        this.renderPieces();
        this.renderHighlights();
    }

    clearSelection() {
        this.selectedPiece = null;
        this.validMoves = [];
        this.renderPieces();
        this.renderHighlights();
    }

    calculateValidMoves(piece) {
        const slideMoves = getValidSlideMoves(this.boardInfo, piece.x, piece.y);
        const lShapeMoves = getValidLShapeMoves(this.boardInfo, piece.x, piece.y);
        this.validMoves = [...slideMoves, ...lShapeMoves];
    }

    renderHighlights() {
        // 기존 하이라이트 제거
        document.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('highlight-slide', 'highlight-lshape');
        });

        // 새로운 하이라이트 추가
        this.validMoves.forEach(move => {
            const cell = this.getCellElement(move.x, move.y);
            if (cell) {
                if (move.type === 'slide') cell.classList.add('highlight-slide');
                else cell.classList.add('highlight-lshape');
            }
        });
    }

    movePiece(piece, targetX, targetY) {
        const fromX = piece.x;
        const fromY = piece.y;

        // 보드 정보 업데이트
        this.boardInfo[piece.y][piece.x] = null;
        piece.x = targetX;
        piece.y = targetY;
        this.boardInfo[targetY][targetX] = piece;

        // 기보 로깅: 개별 수 기록
        this.gameLog.push({
            player: piece.player,
            from: { x: fromX, y: fromY },
            to: { x: targetX, y: targetY }
        });

        // 이동 완료 후 상태 기록
        this.saveHistoryState();

        // 승리 조건 체크
        if (checkWinCondition(targetX, targetY)) {
            this.handleWin(piece.player);
            this.renderPieces();
            this.renderHighlights();
            return;
        }

        this.clearSelection();
        this.switchTurn();
    }

    handleWin(player) {
        this.isGameOver = true;
        this.statusBar.textContent = `🎉 플레이어 ${player} 승리! 🎉 (아래에서 복기 가능)`;
        this.statusBar.style.color = '#f1c40f';
        this.validMoves = [];
        this.showReplayControls();

        // 딥러닝(MCTS) 대전에서 인간 승리 시 기보 서버 전송
        if (this.gameMode === '1p' && player === this.humanPlayer) {
            const aiModeGroup = document.getElementById('ai-mode-seg');
            const aiMode = aiModeGroup?.dataset.value;
            if (aiMode === 'mcts') {
                this.uploadGameLog(player);
            }
        }
    }

    async uploadGameLog(winner) {
        try {
            const baseUrl = import.meta.env.PROD
                ? 'https://7hehans.duckdns.org:8443'
                : 'http://localhost:8001';

            await fetch(`${baseUrl}/api/game-log`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    winner,
                    total_moves: this.gameLog.length,
                    moves: this.gameLog,
                    timestamp: new Date().toISOString()
                })
            });
            console.log('Game log uploaded successfully.');
        } catch (err) {
            console.warn('Game log upload failed (non-critical):', err.message);
        }
    }

    showReplayControls() {
        if (!this.replayControls) return;
        this.replayControls.classList.remove('hidden');
        this.isReplayMode = true;
        this.currentReplayStep = this.moveHistory.length - 1;
        this.updateReplayUI();
    }

    hideReplayControls() {
        if (!this.replayControls) return;
        this.replayControls.classList.add('hidden');
        this.isReplayMode = false;
    }

    replayPrev() {
        if (this.currentReplayStep > 0) {
            this.currentReplayStep--;
            this.applyReplayState();
        }
    }

    replayNext() {
        if (this.currentReplayStep < this.moveHistory.length - 1) {
            this.currentReplayStep++;
            this.applyReplayState();
        }
    }

    applyReplayState() {
        const state = this.moveHistory[this.currentReplayStep];
        this.pieces.forEach((piece, i) => {
            // 과거 기록의 위치로 좌표만 덮어씌움 (CSS transition으로 말들이 스르륵 움직임)
            piece.x = state[i].x;
            piece.y = state[i].y;
            this.updatePieceView(piece);
        });
        this.updateReplayUI();
    }

    updateReplayUI() {
        if (this.replayStatus) {
            this.replayStatus.textContent = `${this.currentReplayStep} / ${this.moveHistory.length - 1}`;
        }
    }

    exitReplay() {
        this.hideReplayControls();
        this.resetGame();
    }

    async switchTurn() {
        this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        this.updateStatus();

        // 2인용 모드: AI 개입 없음
        if (this.gameMode === '2p') {
            this.isAITurn = false;
            return;
        }

        // 1인용 모드: AI 턴인지 확인
        if (this.currentPlayer === this.aiPlayer) {
            this.isAITurn = true;
            this.statusBar.textContent = 'AI가 생각 중입니다...';

            const aiModeGroup = document.getElementById('ai-mode-seg');
            const aiMode = aiModeGroup?.dataset.value || 'minimax';
            
            const powerSlider = document.getElementById('ai-power-slider');
            const powerValue = parseInt(powerSlider?.value || (aiMode === 'mcts' ? '400' : '3'), 10);
            
            const useML = (aiMode === 'mcts');

            if (useML && !this.isAIReady) {
                // If the turn passes to DL AI, but it's not ready yet, show overlay and pend the task
                this.showLoadingOverlay();
                this.pendingAITask = async () => {
                    const move = await this.fetchAIMove(this.boardInfo, this.aiPlayer, powerValue, useML);
                    this.handleAIResponse(move);
                };
            } else {
                // Direct call
                const move = await this.fetchAIMove(this.boardInfo, this.aiPlayer, powerValue, useML);
                this.handleAIResponse(move);
            }
        } else {
            this.isAITurn = false;
        }
    }

    async fetchAIMove(boardInfo, currentPlayer, powerValue, useML) {
        try {
            const simpleBoard = boardInfo.map(row => 
                row.map(cell => cell ? cell.player : 0)
            );
            const humanPlayer = currentPlayer === 1 ? 2 : 1;
            
            this.updateStatus("AI가 생각 중입니다... ⏳");
            
            const bestMove = await new Promise((resolve, reject) => {
                this.onAIMoveReceived = resolve;
                
                if (this.aiWorker) {
                    this.aiWorker.postMessage({ 
                        board: simpleBoard, 
                        aiPlayer: currentPlayer, 
                        humanPlayer, 
                        powerValue,
                        useMCTS: useML
                    });
                } else {
                    reject(new Error("AI Worker not initialized"));
                }
            });
            
            return bestMove;
        } catch (error) {
            console.error('Error fetching AI move:', error);
            return null;
        }
    }

    handleAIResponse(move) {
        if (move) {
            const targetPiece = this.boardInfo[move.pieceY][move.pieceX];
            setTimeout(() => {
                this.movePiece(targetPiece, move.targetX, move.targetY);
            }, 500);
        } else {
            this.statusBar.textContent = 'AI가 이동할 수 없습니다.';
            this.isGameOver = true;
        }
    }

    resetGame() {
        this.hideReplayControls();
        
        // Read game mode settings from segment groups
        const gameModeGroup = document.getElementById('game-mode-group');
        this.gameMode = gameModeGroup?.dataset.value || '1p';
        
        const turnOrderGroup = document.getElementById('turn-order-seg');
        const turnOrder = turnOrderGroup?.dataset.value || 'first';
        
        if (this.gameMode === '1p') {
            this.humanPlayer = (turnOrder === 'first') ? 1 : 2;
            this.aiPlayer = (turnOrder === 'first') ? 2 : 1;
        }
        
        this.initPieces();
        this.currentPlayer = 1;
        this.selectedPiece = null;
        this.validMoves = [];
        this.isGameOver = false;
        this.isAITurn = false;
        this.gameLog = [];
        this.statusBar.style.color = '';
        this.renderPieces();
        this.renderHighlights();
        this.updateStatus();
        
        // If AI goes first (후공 선택 시), trigger AI move
        if (this.gameMode === '1p' && this.aiPlayer === 1) {
            this.isAITurn = true;
            this.statusBar.textContent = 'AI가 생각 중입니다...';
            
            const aiModeGroup = document.getElementById('ai-mode-seg');
            const aiMode = aiModeGroup?.dataset.value || 'minimax';
            
            const powerSlider = document.getElementById('ai-power-slider');
            const powerValue = parseInt(powerSlider?.value || (aiMode === 'mcts' ? '400' : '3'), 10);
            const useML = (aiMode === 'mcts');
            
            if (useML && !this.isAIReady) {
                this.showLoadingOverlay();
                this.pendingAITask = async () => {
                    const move = await this.fetchAIMove(this.boardInfo, this.aiPlayer, powerValue, useML);
                    this.handleAIResponse(move);
                };
            } else {
                this.fetchAIMove(this.boardInfo, this.aiPlayer, powerValue, useML)
                    .then(move => this.handleAIResponse(move));
            }
        }
    }

    updateStatus(customMessage) {
        if (customMessage) {
            this.statusBar.textContent = customMessage;
        } else if (this.gameMode === '2p') {
            this.statusBar.textContent = `플레이어 ${this.currentPlayer}의 턴입니다`;
        } else if (this.currentPlayer === this.humanPlayer) {
            this.statusBar.textContent = '당신의 턴입니다';
        } else {
            this.statusBar.textContent = 'AI가 생각 중입니다...';
        }
    }
}

// 게임 인스턴스 초기화
document.addEventListener('DOMContentLoaded', () => {
    window.game = new HorseRunGame();
});
