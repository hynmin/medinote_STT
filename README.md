# 의료 상담 STT (Speech-to-Text)

의료 상담 음성을 텍스트로 변환하고 화자를 분리하는 프로젝트

## ✨ 주요 기능

- 🎙️ **Whisper 기반 STT**: 고정확도 한국어 음성 인식
- 👥 **화자 자동 감지**: PyAnnote 기반 자동 화자 구별 (의사/환자)
- 📊 **평가 지표**: WER/CER 자동 계산 및 저장
- 💾 **데이터베이스**: SQLite 기반 변환 결과 및 메타데이터 저장 (추후: MySQL)
- 📋 **CLI 출력**: 터미널에서 결과 및 통계 확인


## 🚀 빠른 시작

### 1. 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate #Linux/Mac: source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# HF 토큰 
set HF_TOKEN=your_token_here       # Windows
export HF_TOKEN=your_token_here    # Mac/Linux

#실행
venv\Scripts\Activate.ps1
venv\Scripts\activate.bat

```

### 2. 사용법

#### 단일 파일 변환
```bash
python main.py audio.mp3
```

#### 데이터베이스 경로 지정
```bash
python main.py audio.mp3 --db-path results.db
```

#### 모델 선택
```bash
# fast (whisper-small) - 기본값, 빠름
python main.py audio.mp3 --model fast

# balanced (whisper-medium) - 균형잡힌 성능
python main.py audio.mp3 --model balanced

# accurate (whisper-large-v3) - 최고 정확도
python main.py audio.mp3 --model accurate
```

#### 화자 분리
```bash
# 화자 수 자동 감지
python main.py audio.mp3 --diarization
```

#### 평가 지표
```bash

python main.py data/audio/doctor_conversation.mp3 --ref-file data/reference.txt --model fast

```

#### 디렉토리 일괄 처리
```bash
python main.py data/audio/ --db-path ./results/batch.db --diarization```
```
#### CLI 출력 예시
```bash

$ python main.py consultation.mp3 --diarization --db-path results.db


[처리 중] consultation.mp3
[모델] whisper-medium (balanced)
[화자 분리] 자동 감지 모드
[오디오 길이] 2분 15초

=== 변환 결과 ===
어디가 불편하세요?
목이 아프고 기침이 계속 나요.
언제부터 그러셨나요?

=== 평가 지표 ===
WER (단어 오류율): 5.2%
CER (문자 오류율): 2.8%
처리 시간: 8.3초
화자 수: 2명

```


### 3. 코드에서 사용

```bash
from stt_engine import MedicalSTT


# STT 엔진 초기화
stt = MedicalSTT(model_type="fast", enable_diarization=True)

# 음성 변환
result = stt.transcribe("audio.mp3")


# 평가 지표 계산 (정답 텍스트 제공)
result = stt.transcribe(
    "audio.mp3",
    reference_text="어디가 불편하세요?"
)

print(f"화자 수: {result['num_speakers']}명")
print(f"WER: {result['metrics']['wer']:.2%}")

```

## 📁 프로젝트 구조

```
stt/
├── config.py # 모델 및 화자 분리 설정
├── stt_engine.py # STT 엔진 (Whisper)
├── diarization.py # 화자 자동 분리 엔진 (PyAnnote)
├── metrics.py # WER/CER 계산
├── db_manager.py # SQLite 데이터베이스 관리
├── main.py # CLI 실행
├── requirements.txt
├── data/
│ ├── audio/ # 입력 오디오 파일
│ ├── reference/ # 정답 텍스트 파일 (평가용)
│ └── output/ # 변환 결과 텍스트
├── results/
│ └── transcriptions.db # 데이터베이스
└── transcriptions.db


```

## 🔧 환경 변수

```bash

# GPU 사용 (기본: CPU)
export USE_GPU=true

# 모델 선택
export STT_MODEL=balanced
```

## 📊 모델 비교

| 모델 | 크기 | 속도 | 정확도 | 추천 |
|------|------|------|--------|------|
| fast | 244MB | ⚡⚡⚡ | ⭐⭐⭐ | 개발 |
| balanced | 769MB | ⚡⚡ | ⭐⭐⭐⭐ | 배포 |
| accurate | 1.5GB | ⚡ | ⭐⭐⭐⭐⭐ | 고품질 |


### GPT 요약정리


## 🧪 테스트
```bash
python -m pytest tests/
# 또는
python tests/test_stt.py
```

