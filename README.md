# 🐎 말 달리자 (Horse Run)

사막 한가운데서 오아시스를 다투는 2인 전략 보드게임 웹 버전 구현체입니다.

## 🌟 주요 기능 (Features)
- **독창적인 이동 규칙**: 장기의 '차(車)'처럼 벽이나 장애물을 만날 때까지 미끄러지는 방식(Slide)과 체스의 '나이트(Knight)' 이동 방식(L-Shape)을 조합한 전략 보드게임입니다.
- **Minimax AI 탑재**: 자바스크립트의 Web Worker를 활용해 메인 렌더링 스레드를 방해하지 않으면서 최적의 수를 계산하는(Alpha-Beta Pruning 적용) 인공지능과 대결할 수 있습니다. 
- **난이도 조절 시스템**: 유저가 직접 AI의 수읽기 깊이(Depth 2~5)를 선택할 수 있습니다.
- **기보 복기(Replay) 시스템**: 게임 종료 후, `이전 수`/`다음 수` 버튼으로 게임의 모든 턴을 되감기/빨리감기 하며 패인과 전술을 분석할 수 있습니다.
- **반응형 웹 지원 (Responsive Design)**: 데스크탑은 물론 모바일 환경에서도 브라우저 깨짐 없이 완벽한 비율로 동작합니다.
- **Docker 기반 간편 배포**: Nginx 환경으로 빌드되어 손쉽게 컨테이너 형태로 운영 서버에 배포할 수 있습니다.

## 🛠 기술 스택 (Tech Stack)
- **Frontend**: HTML5, CSS3(Grid, Flexbox), Vanilla JavaScript (ES6+ Modules)
- **Build Tool**: Vite
- **Web Worker**: AI 연산 병렬 처리
- **Deployment**: Docker, Docker Compose, Nginx

## 🚀 로컬 실행 방법 (How to run locally)
Node.js(v16 이상 권장)가 설치되어 있어야 합니다.
```bash
# 의존성 패키지 설치
npm install

# 로컬 개발 서버 실행
npm run dev
```

## 🐳 Docker를 이용한 서버 배포 방법
오라클 클라우드 우분투 서버 등에 포팅할 때 사용합니다.
```bash
# Docker Compose 백그라운드 구동 (자동 빌드 포함)
docker compose up --build -d
```
이후 서버 IP 주소의 8080 포트(예: `http://193.x.x.x:8080`)로 접속하면 게임 플레이가 가능합니다.

## 📜 라이선스 (License)
이 프로젝트는 **MIT License**를 따릅니다. 누구나 자유롭게 코드를 수정하고 배포하실 수 있습니다.
