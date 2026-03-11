# 🐎 말 달리자 (Horse Run)

[![Play Now](https://img.shields.io/badge/Play_Now-GitHub_Pages-success?style=for-the-badge&logo=googlechrome)](https://7hehans.github.io/horse-run-game/)

사막 한가운데서 오아시스를 다투는 2인 전략 보드게임 웹 버전입니다.

## 🌟 주요 기능

- **독창적인 이동 규칙**: 장기의 '차(車)'처럼 벽이나 장애물을 만날 때까지 미끄러지는 슬라이드(Slide)와 체스의 나이트(Knight) L자 이동을 조합한 전략 보드게임
- **자유로운 AI 난이도 조절 시스템 (성능 슬라이더)**
  - ⚙️ **미니맥스 모드**: 깊이(Depth) 2~5 단계 세밀 조절 가능
  - 🧠 **딥러닝 모드 (MCTS + ONNX)**: 브라우저 내부에서 하드웨어 가속(WebGL/WebGPU)을 사용하여 MCTS 반복 횟수를 100~2,000회까지 유저 기기 성능에 맞춰 조절 가능
- **완전한 클라이언트 사이드 (Serverless AI)**: 오라클 서버에 의존하지 않고 사용자의 브라우저 Web Worker 단에서 무거운 딥러닝 추론을 병렬 처리하여 UI 프리징 방지
- **게임 모드**: 👤 1인용(vs AI) / 👥 2인용(로컬 대전)
- **선·후공 선택**: 🔴 선공(빨강) 또는 🟣 후공(보라) 선택 가능
- **기보 복기 시스템**: 게임 종료 후 ◀/▶ 버튼으로 모든 턴을 되감기/빨리감기하며 전술 분석
- **반응형 디자인**: 데스크탑·모바일 환경 모두 지원
- **모던 UI**: 세그먼트 컨트롤, glassmorphism 카드, 3D 입체 게임 말

## 🎮 게임 방법

### 보드 구성
- **11×11** 격자 보드
- 🟡 **사막**: 일반 이동 칸
- 🟢 **초원**: 중앙 주변 십자형 배치 (L자 이동으로 진입 불가)
- 🔵 **오아시스**: 정중앙 (5,5) — **여기에 슬라이드로 도달하면 승리!**

### 이동 규칙
1. **슬라이드**: 상하좌우로 벽이나 다른 말에 닿을 때까지 직진 (초원·오아시스 통과 가능)
2. **L자 이동**: 체스 나이트처럼 이동 (도착지가 **사막**이어야 함)

### 승리 조건
자신의 말 중 하나를 오아시스(🔵)에 **슬라이드**로 정확히 도달시키면 승리!

> 💡 **TIP**: L자 이동으로는 오아시스에 진입할 수 없습니다. 초원을 경유하는 슬라이드 경로를 확보하세요!

## 🛠 기술 스택

| 영역 | 기술 |
|------|------|
| **Frontend** | HTML5, CSS3 (Grid, Flexbox, glassmorphism), Vanilla JS (ES6+ Modules) |
| **AI (Client)** | Minimax, **MCTS**, **ONNX Runtime Web (WebGL/WebGPU/WASM)**, Web Worker |
| **Build Tool** | Vite |
| **Backend** | Python FastAPI (기보 로그 수집 및 초기 모델 학습용) |
| **Deployment** | GitHub Pages (전체 게임 및 AI 구동) |

## 🚀 로컬 실행 방법

Node.js(v16 이상)가 설치되어 있어야 합니다.

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 딥러닝 AI 모델 로컬 학습 및 로깅 서버 (선택사항)
이제 가장 높은 성능을 요구하는 딥러닝 AI 마저도 로컬 브라우저에서(ONNX Runtime Web을 통해) 100% 구동되므로 일반적인 플레이 시 백엔드 서버가 필요하지 않습니다.
다만, 새로운 기보 데이터를 수집하거나 PyTorch 모델 학습 파이프라인을 돌려보고자 할 경우에만 아래와 같이 구동합니다.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python export_onnx.py # 모델 변환 필요 시
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 📚 문서

| 문서 | 내용 |
|------|------|
| [`docs/rules.md`](docs/rules.md) | 게임 규칙 상세 설명 |
| [`docs/STRATEGY.md`](docs/STRATEGY.md) | 전략 가이드 |
| [`docs/architecture.md`](docs/architecture.md) | 시스템 아키텍처 및 파일 구조 |
| [`docs/ai_training_pipeline.md`](docs/ai_training_pipeline.md) | AI 학습 파이프라인 (V1→V6 진화 과정) |
| [`docs/development_history.md`](docs/development_history.md) | Phase별 개발 히스토리 |

## 📜 라이선스

이 프로젝트는 **MIT License**를 따릅니다.
