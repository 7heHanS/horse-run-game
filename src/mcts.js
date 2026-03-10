import { BOARD_SIZE, OASIS_POS } from './constants.js';
import { getValidSlideMoves, getValidLShapeMoves, checkWinCondition } from './engine.js';
import { evaluateBoardV7 } from './heuristics.v7.js';

class MCTSNode {
    constructor(state, player, priorProb, parent = null, move = null) {
        this.state = state; // 2D array
        this.player = player;
        this.priorProb = priorProb;
        this.parent = parent;
        this.move = move;
        
        this.children = new Map(); // key: string "px,py->tx,ty", value: MCTSNode
        this.visitCount = 0;
        this.valueSum = 0.0;
    }

    get qValue() {
        if (this.visitCount === 0) return 0;
        return this.valueSum / this.visitCount;
    }

    expand(actionProbs) {
        // actionProbs: Array of { move, prob }
        for (const { move, prob } of actionProbs) {
            const nextState = this.simulateMove(this.state, move);
            const nextPlayer = this.player === 1 ? 2 : 1;
            const moveKey = `${move.pieceX},${move.pieceY}->${move.targetX},${move.targetY}`;
            
            if (!this.children.has(moveKey)) {
                this.children.set(moveKey, new MCTSNode(
                    nextState,
                    nextPlayer,
                    prob,
                    this,
                    move
                ));
            }
        }
    }

    simulateMove(board, move) {
        const nextBoard = board.map(row => [...row]);
        const p = nextBoard[move.pieceY][move.pieceX];
        nextBoard[move.pieceY][move.pieceX] = 0;
        nextBoard[move.targetY][move.targetX] = p;
        return nextBoard;
    }

    isExpanded() {
        return this.children.size > 0;
    }
}

export class MCTS {
    constructor(onnxSession, ort, cPuct = 2.5, numSimulations = 100) {
        this.session = onnxSession;
        this.ort = ort;
        this.cPuct = cPuct;
        this.numSimulations = numSimulations;
        this.nnCache = new Map(); // key: JSON.stringify(board)+player
    }

    async getActionProbs(board, player) {
        const stateKey = JSON.stringify(board) + player;
        if (this.nnCache.has(stateKey)) return this.nnCache.get(stateKey);

        const possibleMoves = this.getAllPossibleMoves(board, player);
        if (possibleMoves.length === 0) return [];

        // Prepare ONNX input (1, 3, 11, 11)
        const inputData = this.prepareInput(board, player);
        const inputTensor = new this.ort.Tensor('float32', inputData, [1, 3, 11, 11]);
        
        const results = await this.session.run({ input: inputTensor });
        const logitsFrom = results.logits_from.data; // Float32Array(121)
        const logitsTo = results.logits_to.data;     // Float32Array(121)

        // Masking logic
        const fromMask = new Uint8Array(121).fill(0);
        for (const m of possibleMoves) {
            fromMask[m.pieceY * BOARD_SIZE + m.pieceX] = 1;
        }

        // Softmax for From
        const probsFrom = this.softmaxWithMask(logitsFrom, fromMask);
        // Softmax for To (simplified, since we multiply later)
        const probsTo = this.softmax(logitsTo);

        let actionProbs = [];
        for (const move of possibleMoves) {
            const fromIdx = move.pieceY * BOARD_SIZE + move.pieceX;
            const toIdx = move.targetY * BOARD_SIZE + move.targetX;
            const prob = probsFrom[fromIdx] * probsTo[toIdx];
            actionProbs.push({ move, prob });
        }

        // Normalize
        const totalProb = actionProbs.reduce((sum, item) => sum + item.prob, 0);
        if (totalProb > 0) {
            actionProbs = actionProbs.map(item => ({ ...item, prob: item.prob / totalProb }));
        }

        this.nnCache.set(stateKey, actionProbs);
        return actionProbs;
    }

    prepareInput(board, currentPlayer) {
        const opponent = currentPlayer === 1 ? 2 : 1;
        const data = new Float32Array(3 * 11 * 11);
        
        const MEADOW_SET = new Set([
            "5,4", "5,6", "4,5", "6,5",
            "5,3", "5,7", "3,5", "7,5",
            "4,4", "6,4", "4,6", "6,6"
        ]);

        for (let y = 0; y < 11; y++) {
            for (let x = 0; x < 11; x++) {
                const val = board[y][x];
                // Channel 0: Current player
                if (val === currentPlayer) data[0 * 121 + y * 11 + x] = 1.0;
                // Channel 1: Opponent
                else if (val === opponent) data[1 * 121 + y * 11 + x] = 1.0;
                
                // Channel 2: Special tiles
                if (x === 5 && y === 5) data[2 * 121 + y * 11 + x] = 1.0;
                else if (MEADOW_SET.has(`${x},${y}`)) data[2 * 121 + y * 11 + x] = 0.5;
            }
        }
        return data;
    }

    softmaxWithMask(logits, mask) {
        const result = new Float32Array(logits.length);
        let maxVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (mask[i] && logits[i] > maxVal) maxVal = logits[i];
        }
        
        let sumExp = 0;
        for (let i = 0; i < logits.length; i++) {
            if (mask[i]) {
                result[i] = Math.exp(logits[i] - maxVal);
                sumExp += result[i];
            } else {
                result[i] = 0;
            }
        }
        for (let i = 0; i < logits.length; i++) {
            result[i] /= sumExp;
        }
        return result;
    }

    softmax(logits) {
        const result = new Float32Array(logits.length);
        let maxVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) maxVal = logits[i];
        }
        let sumExp = 0;
        for (let i = 0; i < logits.length; i++) {
            result[i] = Math.exp(logits[i] - maxVal);
            sumExp += result[i];
        }
        for (let i = 0; i < logits.length; i++) {
            result[i] /= sumExp;
        }
        return result;
    }

    getAllPossibleMoves(board, player) {
        const moves = [];
        for (let y = 0; y < BOARD_SIZE; y++) {
            for (let x = 0; x < BOARD_SIZE; x++) {
                if (board[y][x] === player) {
                    const slide = getValidSlideMoves(board, x, y);
                    const lshape = getValidLShapeMoves(board, x, y);
                    for (const m of [...slide, ...lshape]) {
                        moves.push({
                            pieceX: x, pieceY: y,
                            targetX: m.x, targetY: m.y,
                            type: m.type, player: player
                        });
                    }
                }
            }
        }
        return moves;
    }

    evaluateNode(board, evalPlayer) {
        const opponent = evalPlayer === 1 ? 2 : 1;
        const rawScore = evaluateBoardV7(board, evalPlayer, opponent);
        return Math.tanh(rawScore / 2000.0);
    }

    async search(initialBoard, player) {
        const root = new MCTSNode(initialBoard, player, 1.0);
        
        // Initial expansion
        const actionProbs = await this.getActionProbs(initialBoard, player);
        root.expand(actionProbs);

        if (!root.isExpanded()) return null;

        for (let sim = 0; sim < this.numSimulations; sim++) {
            let node = root;
            const searchPath = [node];

            // 1. Selection
            while (node.isExpanded()) {
                let bestScore = -Infinity;
                let bestChild = null;
                
                for (const child of node.children.values()) {
                    const q = -child.qValue;
                    const u = this.cPuct * child.priorProb * Math.sqrt(node.visitCount) / (1 + child.visitCount);
                    const score = q + u;
                    if (score > bestScore) {
                        bestScore = score;
                        bestChild = child;
                    }
                }
                node = bestChild;
                searchPath.push(node);
            }

            // 2. Expansion / Evaluation
            let isTerminal = false;
            let value = 0.0;
            
            if (node.move && checkWinCondition(node.move.targetX, node.move.targetY)) {
                isTerminal = true;
                value = -1.0;
            }
            
            if (!isTerminal) {
                const probs = await this.getActionProbs(node.state, node.player);
                if (probs.length === 0) {
                    value = -1.0;
                } else {
                    node.expand(probs);
                    value = this.evaluateNode(node.state, node.player);
                }
            }

            // 3. Backpropagation
            for (let i = searchPath.length - 1; i >= 0; i--) {
                const n = searchPath[i];
                n.valueSum += value;
                n.visitCount += 1;
                value = -value;
            }
        }

        // Return best move by visit count
        let bestChild = null;
        let maxVisits = -1;
        for (const child of root.children.values()) {
            if (child.visitCount > maxVisits) {
                maxVisits = child.visitCount;
                bestChild = child;
            }
        }
        return bestChild ? bestChild.move : null;
    }
}
