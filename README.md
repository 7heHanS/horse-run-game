# 🐎 말 달리자 (Horse Run)

[![Play Now](https://img.shields.io/badge/Play_Now-GitHub_Pages-success?style=for-the-badge&logo=googlechrome)](https://7hehans.github.io/horse-run-game/)

사막 한가운데서 오아시스를 다투는 2인 전략 보드게임 웹 버전입니다.

## 🌟 주요 기능

- **독창적인 이동 규칙**: 장기의 '차(車)'처럼 벽이나 장애물을 만날 때까지 미끄러지는 슬라이드(Slide)와 체스의 나이트(Knight) L자 이동을 조합한 전략 보드게임
- **5단계 AI 난이도 시스템**
  - 😊 **초보** ~ 🧠 **마스터**: Minimax + Alpha-Beta Pruning (2~5수 예측)
  - 👑 **그랜드마스터**: **브라우저 로컬 딥러닝 (ONNX + MCTS)** — 사용자 기기에서 직접 2,000회 수읽기 수행
- **게임 모드**: 👤 1인용(vs AI) / 👥 2인용(로컬 대전)
- **선·후공 선택**: 🔴 선공(빨강) 또는 🟣 후공(보라) 선택 가능
- **Web Worker 기반 AI**: Minimax 연산을 별도 스레드에서 처리하여 UI 프리징 없이 부드러운 게임 플레이
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
| **AI (Client)** | Minimax + Alpha-Beta, **MCTS (Monte Carlo Tree Search)**, **ONNX Runtime Web** |
| **Build Tool** | Vite |
| **Backend** | (선택사항) 기보 로그 수집용 FastAPI 서버 |
| **Deployment** | GitHub Pages (전체 게임 및 AI 구동), Oracle Cloud (데이터 수집용) |

## 🚀 로컬 실행 방법

Node.js(v16 이상)가 설치되어 있어야 합니다.

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 딥러닝 AI 서버 (선택사항)
그랜드마스터 난이도는 이제 브라우저에서 직접 구동되므로 별도의 백엔드 없이도 모든 기능을 사용할 수 있습니다.
 학습이나 로그 수집을 위한 환경 구성은 다음과 같습니다.

```bash
cd backend
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
