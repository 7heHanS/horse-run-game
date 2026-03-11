# 🐎 말 달리자 — 개발 히스토리

프로젝트의 전체 개발 과정을 단계(Phase)별로 정리합니다.

---

## Phase 1: 아키텍처 전환 준비 및 백엔드 서버 구성
- Python FastAPI 기반 백엔드 서버 스켈레톤 작성
- 프론트엔드(`src/main.js`, `src/engine.js`)와 백엔드 API 연동 구조 개발
- 게임 상태 및 좌표 정보를 주고받는 REST API 엔드포인트 구현
- 서버에 기존 Minimax 알고리즘 임시 적용 및 검증

## Phase 2: GitHub Pages 호스팅 및 배포 파이프라인
- 정적 파일 배포를 위한 GitHub Actions 워크플로우 세팅
- 로컬(`localhost:8001`) / 프로덕션(오라클 클라우드) 환경에 따른 API URL 분리
- GitHub Pages 배포 및 오라클 클라우드 서버와의 정상 통신 확인

## Phase 3: 데이터셋 생성 (V1)
- 게임 엔진을 활용한 Minimax Self-play 시뮬레이터 구축
- 대량의 게임 기보(상태-행동 쌍) 추출 (CSV 포맷)
- 대규모 데이터셋 생성 수행

## Phase 4: 딥러닝 모델 V1/V2
- PyTorch 기반 MLP 모델 설계 (Flat board → to-action)
- 기보 데이터셋을 이용한 모델 학습 로직 구현
- 자동 평가 스크립트 (Minimax vs 딥러닝) 및 난이도 검증

## Phase 5: 오라클 클라우드 배포
- 검증된 모델 가중치를 오라클 서버로 SCP 배포
- 백엔드에서 딥러닝 모델 기반 추론 응답 처리
- HTTPS (DuckDNS + Nginx) 적용

## Phase 6: 보안 및 코드 정리
- 불필요한 설정 파일 삭제
- Nginx 설정 템플릿화로 도메인 정보 노출 최소화
- `.gitignore` 정비 (대용량 파일 제외)

## Phase 7: AI 모델 고도화 (Dataset 확장)
- 멀티스레드 기반 기보 데이터셋 생성 스크립트 작성 (32코어 활용)
- 대용량 기보 데이터 활용 모델 재학습

## Phase 8: AI Model V3 — CNN 기반 전면 개선
- **3채널 CNN 아키텍처** 도입 (내 말 / 상대 말 / 지형 맵)
- 14,641개 클래스 출력 (From-To action space)
- **Legal Move Masking** 적용 (학습 + 추론 시 불법 수 완전 제거)
- 데이터셋 개선: 승자 필터링 + `is_critical` 플래그

## Phase 9: AI Model V4 — Multi-Head Architecture
- **Shared Backbone + From Head + To Head** 분리 설계
- From Mask / To Mask 세분화
- Loss Balancing (From 0.5 + To 1.0)
- Minimax Depth 4 대비 압도적 승률 달성

## Phase 11: V5 Fine-Tuning (Overnight MCTS)
- `generate_mcts_dataset.py`: V4+MCTS 3000/5000 자가 대국
- 야간 무중단 가동으로 약 2만 기보 수집
- `is_critical` 가중치 50x 적용 Fine-tuning
- **Pure NN (MCTS 없는 순수 추론)** 으로도 Minimax Depth 4 격파

## Phase 12: ONNX 브라우저 배포 시도 (취소)
- ONNX 변환 및 `onnxruntime-web` 기반 클라이언트 사이드 추론 시도
- WASM 동적 로딩 문제 및 브라우저 호환성 이슈로 **취소**

## Phase 13: 하이브리드 아키텍처 전환
- **Client-side Minimax**: Python AI를 JavaScript로 완전 이식
- **Server-side Pure V5**: 오라클 클라우드에서 딥러닝 추론 전용 API
- `engine.js` 보드 호환성 수정 (`null`/`0` 통합)
- **Web Worker** 적용으로 Depth 3~5에서도 UI 프리징 제거

## Phase 13.5: UI/UX 현대화
- 드롭다운 → **세그먼트 컨트롤 (Pill Buttons)** 전환
- 가로 3단 반응형 레이아웃 + 이모지 아이콘
- 5단계 AI 난이도: 😊 초보 ~ 👑 그랜드마스터(딥러닝)
- 게임 모드 추가: 👤 1인용(vs AI) / 👥 2인용
- 선·후공 선택 기능
- **3D 입체 게임 말**: radial-gradient + inset shadow

## Phase 14: 국지전 집중 훈련 (V6)
- 8-Fold 대칭 증강 로직을 포함한 시드 추출 및 targeted dataset 생성
- **V6 모델 Fine-tuning**: 사이드 공격 방어 성능 25배 강화
- 승리 기보 로그 서버 전송 및 수집 기능 구현

## Phase 15-16: V7 전략적 수비 (Axis Patrol)
- **Stopper-Win** 경로 탐지 로직 구현 (`evaluate_board`)
- **Axis Patrol**: 중앙 축(5행/5열) 점유 보상 체계 도입
- **Axis Cluster**: 상대방의 군집 배치(빌드업) 사전 차단 페널티 적용
- MCTS 시뮬레이션 최적화 (Grandmaster 2000회)

## Phase 17: 규칙 정교화 및 오류 수정
- 마스터(Lv 5) 모드의 초원(Meadow) 침범 버그 수정 (Web Worker 상수 수정)
- 백엔드 `ai.py` 안정화 및 정합성 테스트

## Phase 18: 브라우저 로컬 그랜드마스터 (ONNX Migration)
- **Serverless AI**: 오라클 서버의 추론 부하를 없애기 위해 MCTS/NN 엔진 브라우저 이식
- PyTorch 모델(`.pt`) → **ONNX**(`.onnx`) 변환 및 배포
- **WASM 가속**: `onnxruntime-web`을 통한 고성능 로컬 추론
- **MCTS/Heuristic JS 포팅**: 파이썬 엔진을 자바스크립트로 완전 재구현
- 최종 결과: 모든 난이도가 사용자 기기에서 독립적으로 구동되는 **Full Client-side AI** 완성 (Grandmaster: MCTS 2,000회)

## Phase 19: 성능 슬라이더 UI 및 하드웨어 가속 최적화
- **동적 난이도 조절 (Sliders)**: 기존 5단계의 고정 난이도 버튼을 폐지하고, 기기 성능에 맞춰 연산량을 자유자재로 설정할 수 있는 슬라이더 UI (100~2000 Iterations / Depth 2~5) 도입
- **Hardware Acceleration (ONNX-Web)**: Web Worker 내부 환경 제약을 극복하고 `['webgpu', 'webgl', 'wasm']` 순서의 Fallback 로직을 적용하여 브라우저 GPU 가속 극대화
- **External Data Merging**: 브라우저 런타임 에러 방지를 위해 ONNX 가중치 파일(`.onnx`)과 외부 데이터(`.onnx.data`)를 하나로 통합 배포하도록 렌더링 파이프라인 정리
