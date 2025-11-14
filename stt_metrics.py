"""
STT 관리 평가 지표 계산 (프로덕션 모니터링용)

주요 기능:
- compute_stt_metrics: Whisper 결과 기반 품질 지표 계산
- Confidence Score: Whisper 신뢰도
- Audio Quality: 오디오 품질 (무음, RMS, 클리핑)
- Word Count: 단어 수
"""
import numpy as np
from typing import Dict, List, Any
import librosa


def compute_stt_metrics(
    audio_path: str,
    whisper_output: Dict[str, Any],
    text: str
) -> Dict[str, float]:
    """
    STT 관리 지표 계산

    Args:
        audio_path: 오디오 파일 경로
        whisper_output: Whisper 원본 출력 (segments 포함)
        text: 변환된 텍스트

    Returns:
        관리 지표 딕셔너리
    """
    metrics = {}

    # 1. Confidence Score (Whisper 신뢰도)
    confidence_metrics = _compute_confidence(whisper_output)
    metrics.update(confidence_metrics)

    # 2. Audio Quality (오디오 품질)
    audio_metrics = _compute_audio_quality(audio_path)
    metrics.update(audio_metrics)

    # 3. 단어 수만 계산 (반복/환청 지표 제거)
    metrics["word_count"] = len(text.split())

    return metrics


def _compute_confidence(whisper_output: Dict[str, Any]) -> Dict[str, float]:
    """
    Whisper 신뢰도 지표 계산

    Returns:
        - avg_confidence: 평균 신뢰도
        - min_confidence: 최소 신뢰도
        - low_confidence_ratio: 낮은 신뢰도 비율 (< 0.7)
    """
    segments = whisper_output.get("segments", [])

    if not segments:
        return {
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "low_confidence_ratio": 1.0
        }

    # 각 segment의 confidence 추출 (avg_logprob 사용)
    confidences = []
    for seg in segments:
        # avg_logprob을 확률로 변환 (대략적)
        # avg_logprob은 보통 -1.0 ~ 0.0 범위
        logprob = seg.get("avg_logprob", -1.0)
        # 간단한 변환: exp(logprob) 사용
        confidence = np.exp(logprob)
        confidences.append(confidence)

    avg_conf = float(np.mean(confidences))
    min_conf = float(np.min(confidences))
    low_conf_count = sum(1 for c in confidences if c < 0.7)
    low_conf_ratio = low_conf_count / len(confidences) if confidences else 0.0

    return {
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "low_confidence_ratio": low_conf_ratio
    }


def _compute_audio_quality(audio_path: str) -> Dict[str, float]:
    """
    오디오 품질 지표 계산

    Returns:
        - silence_ratio: 무음 비율
        - audio_rms_energy: RMS 에너지 (SNR 대용)
        - clipping_detected: 클리핑 감지 (0 or 1)
    """
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=None)

        # 무음 비율 계산
        rms = librosa.feature.rms(y=y)[0]
        silence_threshold = 0.01
        silence_frames = np.sum(rms < silence_threshold)
        silence_ratio = silence_frames / len(rms) if len(rms) > 0 else 0.0

        # RMS 에너지 계산
        audio_rms_energy = float(np.sqrt(np.mean(y**2)))

        # 클리핑 감지 (절대값이 0.99 이상인 샘플)
        clipping_threshold = 0.99
        clipping_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_detected = 1 if clipping_samples > len(y) * 0.001 else 0  # 0.1% 이상이면 클리핑

        return {
            "silence_ratio": float(silence_ratio),
            "audio_rms_energy": audio_rms_energy,
            "clipping_detected": clipping_detected
        }

    except Exception as e:
        print(f"⚠️ Audio quality calculation failed: {e}")
        return {
            "silence_ratio": 0.0,
            "audio_rms_energy": 0.0,
            "clipping_detected": 0
        }


# 이전 반복 점수, 환청 위험도 계산 함수 제거됨


# Example usage
if __name__ == "__main__":
    # 테스트용 예시
    whisper_output = {
        "segments": [
            {"avg_logprob": -0.3, "text": "안녕하세요"},
            {"avg_logprob": -0.5, "text": "어디가 불편하세요"},
        ]
    }

    metrics = compute_stt_metrics(
        audio_path="data/audio/test.mp3",
        whisper_output=whisper_output,
        text="안녕하세요 어디가 불편하세요"
    )

    print("\n📊 STT Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
