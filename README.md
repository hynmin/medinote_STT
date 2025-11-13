# 의료 상담 STT (Speech-to-Text)

의료 상담 음성을 텍스트로 변환하고 AI 요약을 생성하는 프로젝트

## ✨ 주요 기능

- 🎙️ **Whisper 기반 STT**: 고정확도 한국어 음성 인식
- 🤖 **AI 요약**: GPT-4o-mini 기반 의료 상담 요약 (주요 증상, 진단, 처방, 생활관리)
- 🔊 **노이즈 제거**: noisereduce 라이브러리 기반 전처리
- 🎯 **VAD 필터**: Voice Activity Detection으로 음성 구간만 처리
- 💾 **SQLite 저장**: 변환 결과 및 메타데이터 저장 (향후: PostgreSQL)
        - 요약정리(탈퇴시까지)
        - 전체텍스트(7일 - 성능테스트용도)
- ☁️ **AWS 연동 준비**: S3 음성 파일 저장(7일), EC2 배포 예정


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
sound_to_text/
├── main.py              # CLI 실행
├── record.py            # 마이크 녹음 (로컬 테스트)
├── stt_engine.py        # STT 엔진 (Whisper)
├── summary.py           # AI 요약 (GPT-4o-mini)
├── storage.py           # SQLite 저장
├── metrics.py           # 품질 평가
├── config.py            # 설정
├── requirements.txt
├── .env                 # 환경 변수 (OpenAI API Key)
└── data/
    ├── audio/           # 테스트용 오디오 파일
    ├── recordings/      # 녹음 파일 (임시, 향후 S3)
    ├── output/          # 변환 결과
    │   └── transcripts.db  # SQLite 데이터베이스
    └── reference.txt    # 평가용 참조 텍스트
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

### 오류 해결
**FFmpeg/torchcodec 문제:**
- `pip uninstall torchcodec` 실행
- transformers가 자동으로 librosa fallback 사용

## 🗺️ 로드맵

### 현재 (로컬 개발)
- ✅ STT 엔진 (Whisper)
- ✅ AI 요약 (GPT-4o-mini)
- ✅ SQLite 저장
- ✅ 노이즈 제거 & VAD 필터

### 다음 단계

#### 🎙️ 녹음 기능
- **현재:** CLI 기반 녹음 (python record.py)
  - 로컬 테스트용
  - `data/recordings/` 임시 저장
- **향후:** React Native WebView + FastAPI
  - 버튼 클릭 녹음
  - S3 직접 업로드
  - JavaScript/HTML 추가

#### ☁️ 인프라
- [ ] AWS S3 연동 (오디오 파일 7일 보관)
- [ ] FastAPI 서버 (녹음 → S3 → STT)
- [ ] PostgreSQL 마이그레이션
- [ ] AWS EC2 배포