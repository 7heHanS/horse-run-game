# 🧠 말 달리자 — AI 학습 파이프라인

## 모델 진화 과정

```
V1 (MLP)  →  V2 (개선 MLP)  →  V3 (CNN)  →  V4 (Multi-Head CNN)  →  V5 (MCTS Fine-tuning)  →  V6 (Targeted Training)  →  V7 (Strategic Heuristics)  →  V8 (ONNX Migration)
```

---

## V1/V2: MLP 기반

- **입력**: Flat board (121칸) + 현재 플레이어 = 122 features
- **출력**: 121개 to-action (도착지만 예측)
- **문제점**: 공간적 패턴 인식 불가, 불법 수 생성

## V3: CNN 기반 + Legal Move Masking

- **입력**: 3채널 × 11×11 텐서
  - Channel 0: 내 말 위치 (binary)
  - Channel 1: 상대 말 위치 (binary)
  - Channel 2: 지형 맵 (오아시스=1.0, 초원=0.5, 사막=0.0)
- **출력**: 14,641 클래스 (121 from × 121 to)
- **Legal Move Masking**: 불법 수의 logit을 −∞로 설정
- **개선**: 승자 기보만 학습 + `is_critical` 플래그

## V4: Multi-Head Architecture

- **Shared Backbone**: Conv2d layers (공유)
- **From Head**: 121개 출력 — 어떤 말을 움직일지
- **To Head**: 121개 출력 — 어디로 보낼지
- **Masking**: From Mask (움직일 수 있는 말) + To Mask (해당 말의 유효 도착지)
- **Loss**: `0.5 × Loss_from + 1.0 × Loss_to`

## V5: MCTS Overnight Fine-tuning

- **Teacher**: V4 + MCTS 3000/5000 자가 대국
- **데이터**: ~2만 기보 (야간 24코어 무중단 생산)
- **학습**: V4 가중치를 base로 Fine-tuning
- **`is_critical` 가중치**: 50x (승자의 마지막 10수에 집중)
- **결과**: 순수 추론(MCTS 없이)으로 Minimax Depth 4 격파

## V6: Targeted Reinforcement (진행 중)

### 취약점
사이드 열 공격(좌/우 열을 타고 중앙으로 슬라이드 진입)에 체계적 무너짐

### 해법: 국지전 집중 훈련
1. 사용자 승리 기보 2개에서 4/6/8턴 시점 보드 상태 추출
2. **8-Fold 대칭 변환**: 90°/180°/270° 회전 × 좌우반전 = 48개 시드
3. V5+MCTS 1500 자가 대국 1,008판 (시드당 21판)
4. 기존 overnight 데이터 + targeted 데이터 합쳐 Fine-tuning
5. targeted 데이터에 3x 가중치 부여

### 학습 설정
| 항목 | 값 |
|------|-----|
| Base Model | `model_v5.pt` |
| Learning Rate | 0.0001 |
| Epochs | 5~10 |
| Critical Weight | 80x |
| Targeted Boost | 3x |

## V7: Strategic Axis Defense (Strategic Heuristics)

- **Axis Patrol**: 중앙 5행/5열 점유 시 강력한 보상 부여 (수비 라인 형성 유도)
- **Stopper-Win Detection**: 다단계 스토퍼 배치를 통한 승리 가능성 사전 감지
- **Axis Cluster Penalty**: 상대방이 특정 축에 말을 쌓는 '필승 대형' 형성 시 페널티 부여
- **효과**: 그랜드마스터 난이도가 단순 근거리 수비가 아닌, 장기적인 전술 배치를 방어함

## V8: ONNX Web Migration (Client-side AI)

- **배경**: 서버 연산 부하 및 지연 시간 해결을 위해 딥러닝 엔진 브라우저 이식
- **변환**: PyTorch `.pt` → ONNX `.onnx` (Opset 12)
- **구현**:
  - `onnxruntime-web` 기반의 비동기 추론 엔진 구축
  - 파이썬 MCTS 로직의 자바스크립트 완전 이식
  - Web Worker 내에서 ONNX-MCTS(2,000회) 병렬 실행

---

## 관련 스크립트

| 파일 | 용도 |
|------|------|
| `generate_dataset_parallel.py` | Minimax self-play 기보 생성 (V1~V3용) |
| `generate_mcts_dataset.py` | V4+MCTS 자가 대국 기보 생성 (V5용) |
| `extract_seed_states.py` | 사용자 기보 → 시드 상태 + 8-fold 증강 |
| `generate_targeted_dataset.py` | 시드 기반 MCTS 자가 대국 (V6용) |
| `train.py` | V1 학습 |
| `train_model_v2.py` | V2 학습 |
| `train_v3.py` | V3 학습 |
| `train_v4.py` | V4 학습 |
| `train_v5.py` | V5 Fine-tuning |
| `train_v6.py` | V6 Targeted Fine-tuning |
| `verify_model.py` | AI 성능 검증 (자동 대전) |
