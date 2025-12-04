"""
STT config
"""
from pathlib import Path
import torch

class STTConfig:
    """STT settings"""

    # HuggingFace 로컬 모델
    HF_MODELS = {
        "fast": "openai/whisper-small",
        "balanced": "openai/whisper-medium",
        "accurate": "openai/whisper-large-v3",
    }

    # OpenAI API 모델
    API_MODELS = {
        "whisper-1": "whisper-1",
        "gpt-4o-transcribe": "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe": "gpt-4o-mini-transcribe",
    }

    # 전체 모델 (개발 테스트용)
    ALL_MODELS = {**HF_MODELS, **API_MODELS}
    MODEL_CHOICES = list(ALL_MODELS.keys())
    DEFAULT_MODEL = "fast"

    LANGUAGE = "korean"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 자동 감지

    # Paths
    AUDIO_DIR = "tests/sample_audio"
    REF_FILE = "tests/reference.txt"

    # Noise handling
    NOISE_REDUCTION = True

    # Silence/short audio thresholds
    MIN_AUDIO_LENGTH = 1.0
    SILENCE_RMS_THRESHOLD = 0.01

    # Initial prompt for Whisper
    INITIAL_PROMPT = "의료 상담 음성입니다. 의료종사자와 환자 대화입니다."

    @classmethod
    def get_model(cls, model_name):
        """모델 경로/이름 반환"""
        if model_name in cls.HF_MODELS:
            return cls.HF_MODELS[model_name]
        elif model_name in cls.API_MODELS:
            return cls.API_MODELS[model_name]
        else:
            raise ValueError(f"Invalid model: {model_name}. Choose from {cls.MODEL_CHOICES}")

    @classmethod
    def is_api_model(cls, name):
        """API 모델인지 확인"""
        return name in cls.API_MODELS

    @classmethod
    def get_device(cls):
        """Return device id (cuda -> 0, cpu -> -1)."""
        return 0 if cls.DEVICE == "cuda" else -1

# Ensure directories exist
Path(STTConfig.AUDIO_DIR).mkdir(parents=True, exist_ok=True)
