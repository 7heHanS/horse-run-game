# 🏗️ 말 달리자 — 시스템 아키텍처

## 전체 구조

```
┌───────────────────────────────────────────────────────────┐
│                    사용자 브라우저                           │
│                                                           │
│  ┌───────────┐      ┌──────────────────────────────────┐  │
│  │  main.js  │      │       minimax.worker.js        │  │
│  │  (UI/UX)  │◀────▶│  (AI 엔진: Minimax & MCTS/ONNX)  │  │
│  └─────┬─────┘      └──────────────────────────────────┘  │
│        │                          ▲                       │
│        │                          │                       │
│        └──────────────────────────┘                       │
│           (모든 난이도 로컬 연산: API 호출 없음)             │
└──────────────────────────┬────────────────────────────────┘
                           │ (선택사항: 게임 로그 전송)
                           ▼
┌───────────────────────────────────────────────────────────┐
│                오라클 클라우드 서버 (Log Server)             │
│                                                           │
│  ┌──────────────┐      ┌─────────────────────────────┐    │
│  │   FastAPI    │      │      game_logs.jsonl        │    │
│  │  (main.py)   │──────│   (기보 데이터 수집/분석)    │    │
│  └──────────────┘      └─────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘
```

## 프론트엔드

| 파일 | 역할 |
|------|------|
| `index.html` | 게임 UI 구조 (세그먼트 컨트롤, 보드, 복기 UI) |
| `src/main.js` | 게임 로직, 이벤트 처리, AI 분기, 복기 시스템 |
| `src/engine.js` | 이동 규칙 엔진 (슬라이드, L자, 승리 판정) |
| `src/constants.js` | 보드 크기, 오아시스/초원 좌표, 초기 배치 |
| `src/minimax.worker.js` | Web Worker — Minimax & **MCTS/ONNX** 통합 엔진 |
| `src/mcts.js` | **Javascript MCTS 포팅** (몬테카를로 트리 탐색 구현) |
| `src/heuristics.v7.js` | **V7 전략적 수비 휴리스틱** (JS 포트) |
| `public/model_v6.onnx` | **딥러닝 모델** (ONNX Web 구동용 가중치) |
| `src/style.css` | glassmorphism, 세그먼트 컨트롤, 3D 게임 말 |

## 백엔드

| 파일 | 역할 |
|------|------|
| `backend/main.py` | FastAPI 서버 — `/api/move` 엔드포인트 |
| `backend/ai.py` | ML 추론 파이프라인 (V4/V5 모델 로딩 및 예측) |
| `backend/engine.py` | Python 게임 엔진 (이동 규칙, 보드 생성) |
| `backend/model_v4.py` | CNN Multi-Head Policy Network 정의 |
| `backend/mcts.py` | Monte Carlo Tree Search 엔진 |

## AI 난이도 시스템

| 난이도 | 방식 | 실행 위치 |
|--------|------|----------|
| 😊 초보 (Depth 2) | Minimax + Alpha-Beta | 브라우저 (Web Worker) |
| ⚔️ 일반 (Depth 3) | Minimax + Alpha-Beta | 브라우저 (Web Worker) |
| 🔥 고수 (Depth 4) | Minimax + Alpha-Beta | 브라우저 (Web Worker) |
| 🧠 마스터 (Depth 5) | Minimax + Alpha-Beta | 브라우저 (Web Worker) |
| 👑 그랜드마스터 | **MCTS (2,000회) + ONNX V6** | 브라우저 (Web Worker) |

## 배포 흐름

```
[로컬 개발] ──npm run dev──→ localhost:3000
      │
      ├── git push (코드만) ──→ GitHub Pages (프론트엔드)
      │
      └── scp model_v6.pt ──→ Oracle Cloud (모델 파일)
                                  └── git pull → uvicorn 재시작
```