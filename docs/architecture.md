/RunHorses
├── /docs
│   └── rules.md          <-- 게임 규칙 명세서
│   └── STRATEGY.md       <-- AI 알고리즘 구현 전략
│   └── architecture.md   <-- 시스템 아키텍처 설계
├── /public               <-- 넷플릭스 캡처 이미지 및 정적 자산
├── /src
│   ├── main.js           <-- Antigravity 앱 초기화 및 렌더링
│   ├── constants.js      <-- 보드 크기, 타일 타입, 초기 배치 좌표
│   ├── engine.js         <-- 이동 규칙(Slide, L-shape) 검증 로직
│   └── ai-worker.js      <-- 미니맥스 알고리즘 (Web Worker)
├── Dockerfile            <-- Nginx 기반 경량 이미지
└── docker-compose.yml