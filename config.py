"""
STT 모델 설정 관리
"""
import os

class STTConfig:
    """STT 모델 설정"""
    
    # 사용 가능한 모델들
    MODELS = {
        "fast": "openai/whisper-small",
        "balanced": "openai/whisper-medium",
        "accurate": "openai/whisper-large-v3"
    }
    
    # 기본 설정
    DEFAULT_MODEL = "fast"  # 개발 단계에서는 fast(small)로 시작
    LANGUAGE = "korean"
    DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

    # 경로 설정
    AUDIO_DIR = "data/audio"
    OUTPUT_DIR = "data/output"

    # 노이즈 처리 설정
    NOISE_REDUCTION = False  # 노이즈 제거 전처리 (noisereduce 필요)
    VAD_FILTER = False  # Voice Activity Detection 필터
    VAD_THRESHOLD = 0.5  # VAD 임계값 (0.0~1.0)

    # Initial Prompt (Whisper에게 제공할 힌트)
    INITIAL_PROMPT = "의료 상담 대화 녹음입니다. 의료종사자와 환자의 대화만 변환하세요."
    
    @classmethod
    def get_model(cls, model_type=None):
        """
        모델 이름 반환
        
        Args:
            model_type: "fast", "balanced", "accurate" 중 하나
                       None이면 환경변수 또는 기본값 사용
        """
        if model_type is None:
            model_type = os.getenv("STT_MODEL", cls.DEFAULT_MODEL)
        
        if model_type not in cls.MODELS:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {list(cls.MODELS.keys())}")
        
        return cls.MODELS[model_type]
    
    @classmethod
    def get_device(cls):
        """디바이스 설정 반환 (cuda/cpu)"""
        return 0 if cls.DEVICE == "cuda" else -1

# 디렉토리 생성
os.makedirs(STTConfig.AUDIO_DIR, exist_ok=True)
os.makedirs(STTConfig.OUTPUT_DIR, exist_ok=True)
