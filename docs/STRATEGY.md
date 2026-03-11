# 🛠 말 달리자 (Horse Run) - 개발 전략 및 아키텍처 (STRATEGY.md)

본 문서는 프로젝트의 기술 스택, 아키텍처 설계, 그리고 AI 알고리즘 구현 전략을 정의합니다.

---

## 1. 시스템 아키텍처 (Architecture)

### 1.1 서버리스 완전 클라이언트 구조 (Serverless Full-Client)
* **Zero-Backend**: 클라우드 서버의 API 호출이나 지연 시간(Latency)을 없애기 위해, 무거운 딥러닝 추론까지 포함한 **모든 게임 및 AI 연산을 브라우저 내부(Web Worker)** 로 완전히 이관하였습니다.
* **로컬 보안**: 서버와 통신하지 않으므로 사용자의 리소스와 화면 렌더링을 온전히 사용자 기기 성능(CPU/GPU)에 맡깁니다.

### 1.2 모듈러 디자인 (Modular Design)
* **View (Main Thread)**: UI 렌더링, 이벤트 리스닝, 3D CSS 및 애니메이션 (Antigravity/Vanilla JS).
* **AI Worker (Background Process)**: 메인 스레드 프리징을 막기 위해 Web Worker에서 ONNX Runtime 모델 및 Minimax 엔진 동작.

---

## 2. AI 알고리즘 전략 (AI Strategy)

### 2.1 미니맥스 + 알파-베타 가지치기 (Minimax with Alpha-Beta Pruning)
* **알고리즘**: 완전 정보 게임에 최적화된 트리 탐색 알고리즘입니다.
* **탐색 깊이(Depth) 동적 조절**: 게임 내 성능 슬라이더를 통해 **2수 ~ 5수 앞**까지 기기 성능에 맞춰 조절 가능합니다.

### 2.2 딥러닝 가속 MCTS (Monte Carlo Tree Search + ONNX)
* **MCTS 시뮬레이션**: PyTorch로 학습시킨 CNN/ResNet 모델(V6)을 ONNX 형태로 포팅하여 JS Web Worker에서 시뮬레이션 확률 밀도를 계산합니다.
* **반복 횟수(Iterations) 동적 조절**: 브라우저 하드웨어 가속(WebGL, WebGPU, WASM)을 활용하여 **100회 ~ 2000회** 폭넓은 연산을 조절합니다.

### 2.2 가중치 기반 상태 평가 함수 (Heuristic Evaluation)
단순 거리가 아닌, 게임의 승리 공식을 수학적으로 모델링합니다.
* **Primary**: `is_win_possible` (현재 위치에서 슬라이드로 오아시스 진입 가능 여부) -> 최상위 가중치
* **Secondary**: `is_on_meadow` (초원 칸 점유) -> 중간 가중치
* **Tertiary**: `blocking_opponent` (상대방의 예상 슬라이드 경로 차단) -> 방어적 가중치

---

## 3. 기술 스택 및 라이브러리 (Tech Stack)

| 구분 | 기술 | 비고 |
| :--- | :--- | :--- |
| **프레임워크** | Antigravity | 경량 웹 개발을 위한 선택 |
| **AI 추론** | ONNX Runtime Web | 브라우저 내 딥러닝 가속 (WASM/WebGL) |
| **AI 알고리즘** | MCTS, Minimax | 난이도별 최적화된 탐색 기법 |
| **멀티스레딩** | Web Worker API | AI 연산 중 브라우저 프리징 방지 |
| **번들러** | Vite | 로컬 개발 생산성 및 빌드 최적화 |
| **배포** | GitHub Pages | 완전한 클라이언트 사이드 서비스 |

---

## 4. 단계별 구현 로드맵 (Roadmap)

### [1단계] 기반 구축 및 UI 렌더링
* 11x11 그리드 생성 및 타일(사막, 초원, 오아시스) 구분.
* 캡처 이미지 기반 초기 말 배치 좌표 설정 및 렌더링.

### [2단계] 게임 엔진 구현 (Core)
* 클릭한 말의 이동 가능 경로 시각화.
* 슬라이드 이동(끝까지 미끄러지기) 및 L자 이동(사막 전용) 로직 완성.

### [3단계] AI 및 1인용 모드
* Web Worker 기반 미니맥스 알고리즘 이식.
* AI의 턴 처리 및 난이도 설정 인터페이스 추가.

### [4단계] 최적화 및 배포
* Docker 빌드 후 오라클 클라우드 이전.
* 모바일 브라우저 호환성 및 터치 조작 최적화.