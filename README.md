# 의료 상담 STT (Speech-to-Text)

의료 상담 음성을 텍스트로 변환하고 AI 요약을 생성하는 프로젝트

## ✨ 주요 기능

- 🎙️ **Whisper 기반 STT**: 고정확도 한국어 음성 인식
- 🤖 **AI 요약**: GPT-4o-mini 기반 의료 상담 요약 (증상, 진단, 권고사항)
- 🔊 **노이즈 제거**: noisereduce 라이브러리 기반 전처리
- 🔇 **무음 감지**: RMS 에너지 기반 환청 방지 (빈 오디오 필터링)
- 💾 **SQLite 저장**: 변환 결과 및 메타데이터 저장 (향후: PostgreSQL)
  - 요약정리 (탈퇴 시까지)
  - 전체텍스트 (7일 - 성능테스트용도)
- ☁️ **AWS 연동 준비**: S3 음성 파일 저장(7일), EC2 배포 예정


## 시작
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
# STT 변환
python tests/test_cli.py tests/sample_audio/consultation.mp3

# 모델 선택
python tests/test_cli.py tests/sample_audio/consultation.mp3 --model fast      # 빠름 (기본값)
python tests/test_cli.py tests/sample_audio/consultation.mp3 --model balanced  # 균형
python tests/test_cli.py tests/sample_audio/consultation.mp3 --model accurate  # 정확

python tests/test_cli.py tests/sample_audio/consultation.mp3 --model whisper-1 
python tests/test_cli.py tests/sample_audio/consultation.mp3 --model gpt-4o-transcribe    # gpt-4o
python tests/test_cli.py tests/sample_audio/consultation.mp3 --model gpt-4o-mini-transcribe  # gpt-4o-mini
```

#### 평가 지표 확인 (개발/테스트용)
```bash
# 참조 텍스트 파일 사용하여 WER/CER 확인
python tests/test_cli.py tests/sample_audio/consultation.mp3 --ref-file tests/reference.txt
```

#### CLI 기반 녹음 테스트용
```bash
python tests/test_record.py
- Space: 녹음 시작/중지
- Enter: STT 처리
- q: 종료
```

## 📁 프로젝트 구조
```bash
sound_to_text/
├── main.py
├── models/
│   └── stt/
│       ├── core/
│       │   └── config.py
│       ├── engine/
│       │   ├── hf_engine.py
│       │   └── openai_engine.py
│       └── pipelines/
│           └── summarize.py
│       └── utils/
│           └── metrics.py
├── db/
│   └── storage.py
├── temp/
│   └── recordings/     ← React 연동시 파일 임시 저장 (S3 업로드 후 삭제)
└── tests/
    ├── test_record.py     ← CLI 녹음 테스트
    ├── test_recordings/   ← CLI 녹음시 파일 저장(개발)
    ├── sample_audio/      ← 테스트용 오디오
    └── reference.txt      ← 평가용 참조 텍스트
```

## 🔧 환경 변수

```bash
프로젝트 루트에 `.env` 파일을 생성하세요:

# 필수: OpenAI API Key (AI 요약용)
OPENAI_API_KEY=your_openai_api_key_here

# 필수: HuggingFace Token (Whisper 모델 다운로드용)
HF_TOKEN=your_huggingface_token_here

# 필수: PostgreSQL Database URL
DATABASE_URL=postgresql://유저:비밀번호@호스트:5432/DB이름
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
- STT 엔진 (HF / openAI)
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


### 로직
STT 로직
[입력]
1) 녹음 버튼을 누른다. 녹음완료버튼을 누르면 모바일에 mp3파일이 생성된다. 
2) 녹음파일(mp3, m4a)을 업로드하는 버튼을 통해 파일 업로드한다

[로컬 임시 저장]
로컬에 임시 파일 저장 (data/temp/20250101_123321_recordingmp3)

[DB 생성]
STT_Transcript테이블에 레코드 생성, 
audio_file(로컬경로), stt_status='pending'
transcript_id 반환

[STT처리 & S3 업로드 병렬 처리]
1) STT 처리
로컬 파일 읽어서 stt.transcribe() 실행. result["text"] 생성
→ STT_Transcript에 transcript_text 저장, processing_time, audio_length, rtf 저장
→ stt_status = 'trascribed' 업데이트
→ generate_summary() 실행 → STT_Summary 저장,
→ stt_status = 'completed' 업데이트

2) S3업로드
→ S3 업로드 성공 : S3_url 저장, S3_status='S3_uploaded' & 로컬 임시파일 삭제 
→ S3 업로드 실패 : S3_status = 'S3_failed' (로컬 파일 유지, 백그라운드 재시도 attempts +=1 업데이트, 제한 3회)

# 할일
- fastapi 연동
- S3 연결( s3_url, full_text )