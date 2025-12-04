"""
OpenAI Whisper API STT 엔진
"""
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed


# 파일 크기 제한
MAX_FILE_SIZE = 25 * 1024 * 1024      # OpenAI API 제한 (25MB) - 초과 시 청크 분할
MAX_UPLOAD_SIZE = 100 * 1024 * 1024   # 업로드 제한 (100MB) - 초과 시 거부
# 청크 길이 (10분 = 600초 = 600,000ms)
CHUNK_LENGTH_MS = 10 * 60 * 1000


class OpenAIWhisperSTT:
    def __init__(self, model="whisper-1"):
        """
        Args:
            model: OpenAI 모델 (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def transcribe(self, audio_path):
        """
        오디오 파일을 OpenAI Whisper API로 변환
        25MB 초과 시 10분 단위로 분할 처리

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            dict: {
                "text": 변환된 텍스트,
                "audio_file": 파일명,
                "model": 모델명,
                "processing_time": 처리시간,
                "audio_length": 오디오 길이
            }
        """
        file_size = os.path.getsize(audio_path)

        # 100MB 초과 시 거부
        if file_size > MAX_UPLOAD_SIZE:
            raise ValueError(f"파일 크기({file_size / 1024 / 1024:.1f}MB)가 100MB 제한을 초과했습니다.")

        if file_size <= MAX_FILE_SIZE:
            return self._transcribe_single(audio_path)
        else:
            print(f"  File size ({file_size / 1024 / 1024:.1f}MB) exceeds 25MB limit. Splitting into chunks...")
            return self._transcribe_chunked(audio_path)

    def _transcribe_single(self, audio_path):
        """단일 파일 STT 처리 (25MB 이하)"""
        print(f"\nProcessing with OpenAI API: {audio_path}")
        start_time = time.time()

        # 모델별 response_format 설정
        if self.model.startswith("gpt-4o"):
            response_format = "json"
        else:
            response_format = "verbose_json"

        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language="ko",
                response_format=response_format
            )

        processing_time = time.time() - start_time

        if response_format == "verbose_json":
            text = response.text
            audio_length = response.duration if hasattr(response, 'duration') else None
        else:
            text = response.text if hasattr(response, 'text') else response.get("text", "")
            audio_length = None

        return {
            "text": text,
            "audio_file": Path(audio_path).name,
            "model": f"openai/{self.model}",
            "processing_time": round(processing_time, 2),
            "audio_length": audio_length,
            "timestamp": datetime.now().isoformat()
        }

    def _process_single_chunk(self, args):
        """단일 청크 STT 처리 (병렬용)"""
        idx, chunk, total = args

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            # 청크 저장
            chunk.export(temp_path, format="mp3")
            print(f"  Processing chunk {idx + 1}/{total}...")

            # API 호출
            if self.model.startswith("gpt-4o"):
                response_format = "json"
            else:
                response_format = "verbose_json"

            with open(temp_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="ko",
                    response_format=response_format
                )

            if response_format == "verbose_json":
                text = response.text
            else:
                text = response.text if hasattr(response, 'text') else response.get("text", "")

            return idx, text
        finally:
            # 임시 파일 삭제
            try:
                os.unlink(temp_path)
            except:
                pass

    def _transcribe_chunked(self, audio_path):
        """청크 분할 STT 처리 (25MB 초과)"""
        print(f"\nProcessing with OpenAI API (chunked): {audio_path}")
        start_time = time.time()

        # 오디오 로드
        file_ext = Path(audio_path).suffix.lower()
        if file_ext == ".mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif file_ext == ".m4a":
            audio = AudioSegment.from_file(audio_path, format="m4a")
        elif file_ext == ".wav":
            audio = AudioSegment.from_wav(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)

        audio_length_sec = len(audio) / 1000  # 밀리초 → 초

        # 10분 단위로 분할
        chunks = []
        for i in range(0, len(audio), CHUNK_LENGTH_MS):
            chunk = audio[i:i + CHUNK_LENGTH_MS]
            chunks.append(chunk)

        print(f"  Split into {len(chunks)} chunks ({CHUNK_LENGTH_MS // 60000} min each)")

        # STT 처리 (병렬 처리)
        chunk_args = [(idx, chunk, len(chunks)) for idx, chunk in enumerate(chunks)]

        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._process_single_chunk, arg): arg[0] for arg in chunk_args}

            for future in as_completed(futures):
                idx, text = future.result()
                results[idx] = text

        # 순서대로 합침
        all_texts = [results[i] for i in range(len(chunks))]

        processing_time = time.time() - start_time
        combined_text = " ".join(all_texts)

        return {
            "text": combined_text,
            "audio_file": Path(audio_path).name,
            "model": f"openai/{self.model}",
            "processing_time": round(processing_time, 2),
            "audio_length": round(audio_length_sec, 2),
            "timestamp": datetime.now().isoformat()
        }

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model": self.model,
            "type": "OpenAI API"
        }
