"""
OpenAI Whisper API STT 엔진
"""
import os
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI


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

        Args:
            audio_path: 오디오 파일 경로

        Returns:
            dict: {
                "text": 변환된 텍스트,
                "audio_file": 파일명,
                "model": 모델명,
                "processing_time": 처리시간
            }
        """
        print(f"\nProcessing with OpenAI API: {audio_path}")
        start_time = time.time()

        # 모델별 response_format 설정
        # gpt-4o-transcribe 모델들은 verbose_json 미지원
        if self.model.startswith("gpt-4o"):
            response_format = "json"
        else:
            response_format = "verbose_json"

        # API 호출
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language="ko",  # 한국어 지정
                response_format=response_format
            )

        processing_time = time.time() - start_time

        # response_format에 따른 결과 파싱
        if response_format == "verbose_json":
            text = response.text
            audio_duration = response.duration if hasattr(response, 'duration') else None
        else:
            # json 포맷은 dict 형태
            text = response.text if hasattr(response, 'text') else response.get("text", "")
            audio_duration = None  # json 포맷은 duration 미제공

        return {
            "text": text,
            "audio_file": Path(audio_path).name,
            "model": f"openai/{self.model}",
            "processing_time": round(processing_time, 2),
            "audio_duration": audio_duration,
            "timestamp": datetime.now().isoformat()
        }

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model": self.model,
            "type": "OpenAI API"
        }
