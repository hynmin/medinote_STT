# 의료 상담 STT (Speech-to-Text)

의료 상담 음성을 텍스트로 변환하고 AI 요약을 생성하는 프로젝트

## ✨ 주요 기능

- 🎙️ **Whisper 기반 STT**: 고정확도 한국어 음성 인식
- 🤖 **AI 요약**: GPT-4o-mini 기반 의료 상담 요약 (주요 증상, 진단, 처방, 생활관리)
- 🔊 **노이즈 제거**: noisereduce 라이브러리 기반 전처리
- 🔇 **무음 감지**: RMS 에너지 기반 환청 방지 (빈 오디오 필터링)
- 💾 **SQLite 저장**: 변환 결과 및 메타데이터 저장 (향후: PostgreSQL)
  - 요약정리 (탈퇴 시까지)
  - 전체텍스트 (7일 - 성능테스트용도)
- ☁️ **AWS 연동 준비**: S3 음성 파일 저장(7일), EC2 배포 예정


## 🚀 빠른 시작

### 1. 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate #Linux/Mac: source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# .env 파일 생성 (필수)
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 2. 사용법

#### 오디오 파일 변환
```bash
# 단일 파일 변환
python main.py data/audio/consultation.mp3

# 모델 선택
python main.py data/audio/consultation.mp3 --model fast      # 빠름 (기본값)
python main.py data/audio/consultation.mp3 --model balanced  # 균형
python main.py data/audio/consultation.mp3 --model accurate  # 정확
```
#### 평가 지표 확인 (개발/테스트용)
```bash
# 또는 참조 텍스트 파일 사용하여 WER/CER 확인
python main.py data/audio/consultation.mp3 --ref-file data/reference.txt
```

#### 현재 녹음기능 : CLI 기반 녹음 (로컬 테스트용)
```bash
python record.py
```
- Space: 녹음 시작/중지
- Enter: STT 처리
- q: 종료

#### CLI 출력 예시
```bash
$ python main.py data/audio/consultation.mp3

🎤 Processing: data/audio/consultation.mp3
  📊 Audio RMS energy: 0.1234 (threshold: 0.05)
  🔧 Applying noise reduction...

==================================================
📄 변환 결과:
==================================================
어디가 불편하세요? 목이 아프고 기침이 계속 나요.

🗄️  Saved to DB: data/output/transcripts.db (transcript_id=1)

⚡ Performance
  RTF: 0.3214 (실시간보다 3.11배 빠름)
  처리 시간: 8.30초 / 오디오 길이: 25.84초

🤖 AI 요약 생성 중...

==================================================
🤖 AI 요약
==================================================

📌 주요 증상:
  목 통증, 지속적인 기침

🏥 진단:
  상기도 감염 의심

💊 약물 처방:
  해열진통제, 기침억제제

🏃 생활 관리:
  - 따뜻한 물 자주 마시기

  ↳ 요약 생성 시간: 2.15초 (summary_id=1)
```


### 3. 코드에서 사용

```python
from stt_engine import MedicalSTT

# STT 엔진 초기화
stt = MedicalSTT(model_type="fast")

# 음성 변환
result = stt.transcribe("audio.mp3")

print(f"변환 텍스트: {result['text']}")
print(f"처리 시간: {result['processing_time']}초")
```

## 📁 프로젝트 구조

```
sound_to_text/
├── main.py              # CLI 실행
├── record.py            # 마이크 녹음 (로컬 테스트)
├── stt_engine.py        # STT 엔진 (Whisper)
├── stt_summary.py       # AI 요약 (GPT-4o-mini)
├── db_storage.py        # DB 저장 (SQLite → PostgreSQL)
├── dev_metrics.py       # 개발 평가지표 (WER/CER/RTF)
├── stt_metrics.py       # 관리 평가지표 (Confidence, Audio Quality)
├── config.py            # 설정
├── requirements.txt
├── .env                 # 환경 변수 (OpenAI API Key, HF Token)
└── data/
    ├── audio/           # 테스트용 오디오 파일
    ├── recordings/      # 녹음 파일 (임시, 향후 S3)
    ├── output/          # 변환 결과
    │   └── transcripts.db  # SQLite 데이터베이스
    └── reference.txt    # 평가용 참조 텍스트
```

## 🔧 환경 변수

프로젝트 루트에 `.env` 파일을 생성하세요:

```bash
# 필수: OpenAI API Key (AI 요약용)
OPENAI_API_KEY=your_openai_api_key_here

# 필수: HuggingFace Token (Whisper 모델 다운로드용)
HF_TOKEN=your_huggingface_token_here
```

## ⚠️ 오류 해결

### FFmpeg/torchcodec 문제
```bash
pip uninstall torchcodec
```
- transformers가 자동으로 librosa fallback 사용
- 이후 화자분리 기능 구현시 pyannote, torchcodec설치

### "Invalid audio file path" 에러
- 파일 경로를 절대 경로 또는 `data/audio/파일명.mp3` 형식으로 지정
- 현재 디렉토리 기준 상대 경로 사용

## 🗺️ 로드맵

### ✅ 현재 (로컬 개발)
- STT 엔진 (Whisper)
- AI 요약 (GPT-4o-mini)
- SQLite 저장
- 노이즈 제거 & 무음 감지
- CLI 녹음 기능(python record.py), `data/recordings/` 임시 저장

### 다음 단계
- FastAPI 서버
- SQLite -> postgreSQL 저장
- React Native WebView + FastAPI 녹음 버튼 녹음 (JavaScript/HTML)
- AWS S3 저장 연동
- AWS EC2 배포

