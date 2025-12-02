FROM python:3.11-slim

WORKDIR /app

# ffmpeg 설치 (pydub 의존성)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p temp/recordings tests/test_recordings db

# 환경변수 (런타임에 오버라이드)
ENV PYTHONUNBUFFERED=1

# 기본 명령어
CMD ["python", "main.py", "--help"]
