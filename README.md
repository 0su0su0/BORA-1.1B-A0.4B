# BORA-1.1B-A0.4B

**EMA + NSA Hybrid Architecture with Sparse MoE**

BORA-1.1B-A0.4B는 **1.1B 파라미터 (366M active)** 규모의 한국어 특화 소형 언어 모델로, EMA(Exponential Moving Average) 기반 시간 믹싱과 NSA(Native Sparse Attention)를 결합한 하이브리드 아키텍처를 사용한다. Apple Silicon (MLX) 환경에서 최적화되어 개발되었다.

---

## 모델 아키텍처

### Core Components

- **Total Layers**: 23 layers
- **Parameter Count**: 1.1B (366M active per token)
- **Hidden Size**: 1024
- **Max Context Length**: Long context support (RoPE θ=1M)
- **Vocabulary**: 32,128 tokens

### Tokenizer

BORA-1.1B-A0.4B는 [VAETKI](https://huggingface.co/NC-AI-consortium-VAETKI/VAETKI)의 토크나이저를 기반으로 최적화된 경량 토크나이저를 사용한다:

- **Base**: VAETKI 토크나이저
- **Optimization**: 불필요한 다국어 토큰 제거
- **Coverage**:
  - 한국어 (완전 지원)
  - 알파벳 대소문자 52자 (A-Z, a-z)
  - 기초 기호 (숫자, 문장부호, 특수문자)
- **Final Vocabulary**: 32,128 tokens
- **Special Tokens**: `<pad>`, `<s>`, `</s>`, `<unk>`

이를 통해 메모리 효율성과 한국어 처리 성능을 동시에 확보하였다

### Layer Pattern (EMA + NSA Attention Hybrid)

```
[EMA, EMA, EMA, NSA, EMA, EMA, EMA, NSA, EMA, EMA, EMA, NSA,
 EMA, EMA, NSA, EMA, EMA, NSA, EMA, NSA, EMA, NSA, EMA]
```

- **EMA Layers (16개)**: MEGA-style Damped EMA 시간 믹싱
  - RWKV에서 영감을 받은 빠른 병렬 스캔
  - Convolution (kernel=4) + Damped EMA
  - Expansion factor: 2x

- **Attention Layers (7개)**: NSA (Native Sparse Attention)
  - GQA (Grouped Query Attention): 8 heads, 4 KV heads
  - Block-level compression: 8 tokens per block
  - Top-k block selection: 16 blocks from 32
  - Sliding window: 512 tokens
  - RoPE θ = 1M (장문맥 지원)

- **MoE FFN**: Sparse Mixture of Experts (DeepSeek-V3)
  - 16 sparse experts + 2 shared experts
  - Top-2 routing per token
  - FFN dimension: 768
  - Sigmoid gate activation

---

## 기술적 특징

### 핵심 아이디어: EMA-NSA Hybrid의 설계 철학

BORA-1.1B-A0.4B는 **EMA의 요약 능력**과 **NSA의 선택적 집중**을 결합하여 효율성과 언어 이해력을 동시에 확보하였다:

#### EMA: 시간적 흐름을 뭉뚱그려 요약
- **Temporal Smoothing**: 과거 정보를 지수 가중 평균으로 압축
- **전역적 맥락 파악**: 긴 범위의 정보를 단일 상태로 요약
- **효율적 처리**: O(n) 복잡도로 메모리/연산 부담 최소화

#### NSA: 중요한 지점만 선택적으로 결합
- **Selective Attention**: Top-k block selection으로 핵심 정보만 추출
- **정밀한 연결**: 중요한 토큰 간 장거리 의존성 포착
- **희소성 활용**: 50% sparsity로 불필요한 연산 제거

#### 조합의 시너지
1. **EMA가 전체 흐름을 요약** → 문맥의 큰 그림 파악
2. **NSA가 핵심 지점을 연결** → 중요한 정보 간 정밀한 관계 학습
3. **결과**:
   - 효율성: EMA의 O(n) + NSA의 희소성 = 메모리 절약
   - 언어 이해: EMA의 요약 + NSA의 선택적 집중 = 품질 유지

---

### 1. MEGA-style Damped EMA
- 기존 RWKV EMA에 damping factor 추가
- State preservation 향상 (δ < 1)
- Padding-aware processing

### 2. NSA (Native Sparse Attention)
- Hardware-friendly block-level sparsity
- Compressed KV cache (8:1 compression ratio)
- Top-k block selection (16/32 blocks)
- Sliding window for locality

### 3. Optimized MoE (DeepSeek-V3)
- Shared experts (2개) for common patterns
- **Per-layer gamma scheduling**: 깊이에 따른 차등 적용
  - DeepSeek-V3는 모든 레이어에 gamma=0.001 단일 값 사용
  - BORA는 레이어 깊이에 따라 발산 정도가 다른 점을 고려
  - **얕은 레이어**: 낮은 gamma (0.0001) - expert 진동 방지
  - **깊은 레이어**: 높은 gamma (0.0005) - 적극적 밸런싱
  - Sqrt 스케줄: 초반 천천히, 후반 빠르게 증가
- Expert health monitoring (Dead/Weak/Healthy detection)

### 4. 장문맥 지원

BORA-1.1B-A0.4B는 EMA-NSA 하이브리드 아키텍처를 통해 메모리 효율적인 장문맥 처리가 가능하다:

#### RoPE θ = 1,000,000 (1M)
- 일반 LLM (θ=10,000)보다 **100배 긴 외삽(extrapolation)** 지원
- 학습된 범위를 넘어서도 위치 임베딩 품질 유지
- 이론적으로 수백만 토큰까지 확장 가능한 설계

#### EMA의 장점 (O(n) 복잡도)
- **Attention의 O(n²) 문제 없음**: 시퀀스 길이에 선형 비례
- **병렬 스캔 알고리즘**: Cumsum trick으로 빠른 학습/추론
- **상태 보존**: Damped EMA로 긴 의존성 포착 가능

#### NSA의 메모리 효율성
- **Compressed KV Cache**: 8 tokens → 1 compressed key/value (8:1 압축)
- **Block-level Sparsity**:
  - 32개 block 중 16개만 선택 (50% sparsity)
  - 중요한 block만 처리하여 메모리 절약
- **Sliding Window**: 512 tokens local attention으로 근거리 context 보존

#### 메모리 효율성 분석

**복잡도 클래스**:
- 일반 Full Attention (23 layers): O(n²)
- BORA (16 EMA + 7 NSA): O(n²) (NSA로 인해)

**실제 메모리 사용량** (상수 계수):
- 일반 Attention: 23 layers × n²
- BORA:
  - EMA (16 layers): 16 × n
  - NSA (7 layers): 7 × n² × 0.5 (sparsity) × 0.125 (8:1 compression)
  - 합계: 16n + 0.4375n²

**결론**: 복잡도 클래스는 동일하지만(O(n²)), EMA 비중(16/23 ≈ 70%)과 NSA의 압축/희소성 덕분에 실제 메모리 사용량이 크게 감소하여 장문맥 처리가 가능하다.

#### 전체 메모리 복잡도

**레이어별 메모리 사용량**:
- **EMA layers (16개)**: O(1) 고정 크기 state
- **NSA layers (7개)**: O(n/8) 압축된 KV cache

**전체 모델 메모리**:
- EMA (16 layers): 상수 (컨텍스트 길이와 무관)
- NSA (7 layers): 7 × (n/8)
- **합계**: 7n/8

**일반 모델과의 비교**:
- 일반 Full Attention (23 layers): 23n
- BORA (16 EMA + 7 NSA): 7n/8
- **감소율**: 7n/8 ÷ 23n ≈ 0.038 (약 3.8%)
- 결과: 일반 모델 대비 **약 26배 적은 KV cache 메모리**

---

## 파일 구성

```
BORA-1.1B-A0.4B/
├── model.py                    # 모델 아키텍처 (EMA + NSA + MoE)
├── config.json                 # 모델 설정
├── train.py                    # 학습 스크립트
├── generate.py                 # 텍스트 생성 (스트리밍)
├── model.safetensors           # 모델 가중치
├── tokenizer.json              # 토크나이저 (32k vocab)
├── tokenizer_config.json
├── special_tokens_map.json
└── samples/                    # 학습 과정 시각화 및 추론 샘플
    ├── train_dashboard.gif     # 학습 대시보드 (애니메이션)
    ├── train_plot.png          # 학습 곡선
    ├── train.log               # 학습 로그
    └── 추론테스트.png           # Step 600 체크포인트 추론 결과
```

### Training Checkpoints

학습 체크포인트는 HuggingFace Hub에서 확인할 수 있다:

**[dororodoroddo/BORA-1.1B-A0.4B-checkpoint](https://huggingface.co/dororodoroddo/BORA-1.1B-A0.4B-checkpoint)**
- Step 600 체크포인트 포함
- 학습 과정 전체 추적 가능

---

## 사용법

### 환경 요구사항
- **Hardware**: Apple Silicon (M1/M2/M3/M4) 또는 Apple GPU
- **Python**: 3.11+
- **MLX**: 최신 버전

### 의존성 설치

```bash
pip install mlx mlx-lm transformers numpy
```

### 텍스트 생성

```bash
python3 generate.py
```

**옵션**:
- 대화형 모드로 실행
- 스트리밍 출력 지원
- 온도, top-p 등 생성 파라미터 조정 가능

---

## PyTorch 포팅 시 주의사항

**경고**: 이 모델은 MLX 프레임워크로 구현되어 있다. **MLX 코드를 PyTorch로 그대로 옮기면 작동하지 않는다.** MLX와 PyTorch는 API와 동작 방식이 다르므로 수동 변환이 필요하다.

### 주요 차이점

1. **파라미터 선언**: MLX는 module attribute를 자동으로 학습 가능하게 처리하지만, PyTorch는 `nn.Parameter()`로 명시적으로 감싸야 함
2. **컴파일**: `mx.compile()` → `torch.compile()` 또는 `@torch.jit.script`로 변경 필요
3. **커스텀 연산**: RoPE, EMA, NSA 등의 구현은 MLX API를 PyTorch API로 수동 변환 필요
4. **메모리 관리**: MLX의 lazy evaluation과 PyTorch의 eager execution 차이 고려 필요

### 예시: 파라미터 선언 실수

```python
# ❌ 잘못된 변환
self.decay_logit = torch.full((dim,), value)  # 학습 안 됨!

# ✓ 올바른 변환
self.decay_logit = nn.Parameter(torch.full((dim,), value))  # nn.Parameter로 명시적 선언 필요
```
---

## 라이선스

Apache 2.0

---

**Author**: 0su0su0
**Framework**: MLX (Apple)
**Architecture**: EMA (MEGA) + NSA + MoE (DeepSeek-V3)
