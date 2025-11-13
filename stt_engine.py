"""
STT 엔진 핵심 로직
"""
from transformers import pipeline
import time
import json
from pathlib import Path
from datetime import datetime
from config import STTConfig
import librosa
import numpy as np
import os


class MedicalSTT:
    """의료 상담 음성을 텍스트로 변환하는 STT 엔진"""

    def __init__(self, model_type="fast", noise_reduction=False):
        """
        Args:
            model_type: "fast", "balanced", "accurate" 중 하나
            noise_reduction: 노이즈 제거 전처리 사용 여부
        """
        self.model_type = model_type
        self.model_name = STTConfig.get_model(model_type)
        self.noise_reduction = noise_reduction

        print(f"🔄 Loading {self.model_name}...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=STTConfig.get_device(),
            return_timestamps=True
        )
        print(f"✅ Model loaded successfully!")
    
    def transcribe(self, audio_path, save_result=True, reference_text=None):
        """
        오디오 파일을 텍스트로 변환

        Args:
            audio_path: 오디오 파일 경로
            save_result: 결과를 JSON으로 저장할지 여부
            reference_text: 평가용 참조 텍스트 (제공 시 WER/CER 계산)

        Returns:
            dict: {
                "text": 변환된 텍스트,
                "audio_file": 오디오 파일명,
                "model": 사용한 모델,
                "processing_time": 처리 시간(초),
                "timestamp": 변환 시각,
                # reference_text 제공 시 추가:
                "metrics": {"wer": 0.05, "cer": 0.02}
            }
        """
        print(f"\n🎤 Processing: {audio_path}")

        start_time = time.time()

        # 오디오 길이 측정
        audio_duration = self._get_audio_duration(audio_path)

        # 오디오 로드하여 에너지 레벨 체크
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 1) 너무 짧은 오디오 체크
        if audio_duration < STTConfig.MIN_AUDIO_DURATION:
            print(f"  🔇 Audio too short ({audio_duration:.1f}s < {STTConfig.MIN_AUDIO_DURATION}s). Returning empty result.")
            processing_time = time.time() - start_time
            return {
                "text": "",
                "audio_file": Path(audio_path).name,
                "model": self.model_name,
                "processing_time": round(processing_time, 2),
                "audio_duration": round(audio_duration, 2),
                "rtf": round(processing_time / max(audio_duration, 0.001), 4)
            }

        # 2) 무음 체크 (RMS 에너지)
        audio_rms = np.sqrt(np.mean(y**2))
        print(f"  📊 Audio RMS energy: {audio_rms:.4f} (threshold: {STTConfig.SILENCE_RMS_THRESHOLD})")
        if audio_rms < STTConfig.SILENCE_RMS_THRESHOLD:
            print(f"  🔇 Audio too quiet. Returning empty result.")
            processing_time = time.time() - start_time
            return {
                "text": "",
                "audio_file": Path(audio_path).name,
                "model": self.model_name,
                "processing_time": round(processing_time, 2),
                "audio_duration": round(audio_duration, 2),
                "rtf": round(processing_time / max(audio_duration, 0.001), 4)
            }

        # STT 수행 (ffmpeg 미설치 시 librosa로 대체 로딩)
        generate_kwargs = {
            "language": STTConfig.LANGUAGE,
            "task": "transcribe"
        }

        # 노이즈 제거 전처리 (이미 로드된 오디오 사용)
        if self.noise_reduction:
            print("  🔧 Applying noise reduction...")
            y = self._apply_noise_reduction(y, sr)
            audio_input = {"array": np.asarray(y), "sampling_rate": sr}
        else:
            audio_input = {"array": np.asarray(y), "sampling_rate": sr}

        # STT 수행 (librosa로 이미 로드했으므로 에러 복구 불필요)
        result = self.transcriber(audio_input, generate_kwargs=generate_kwargs)

        processing_time = time.time() - start_time

        # 결과 정리
        output = {
            "text": result["text"],
            "audio_file": Path(audio_path).name,
            "model": self.model_name,
            "processing_time": round(processing_time, 2),
            "audio_duration": round(audio_duration, 2),
            "timestamp": datetime.now().isoformat()
        }

        # 평가 지표 계산 (옵션)
        if reference_text:
            from metrics import compute_metrics
            metrics = compute_metrics(reference_text, output["text"])
            output["metrics"] = metrics
            print(f"📐 WER: {metrics['wer']:.2%}, CER: {metrics['cer']:.2%}")

        print(f"✅ Done in {processing_time:.2f}s")
        print(f"📝 Result: {output['text'][:100]}...")

        # 결과 저장
        if save_result:
            self._save_result(output)

        return output

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": "GPU" if STTConfig.get_device() == 0 else "CPU",
            "language": STTConfig.LANGUAGE,
            "noise_reduction": self.noise_reduction
        }


# 간단한 사용 예시
if __name__ == "__main__":
    stt = MedicalSTT(model_type="fast")
    result = stt.transcribe("audio.mp3")

    print("\n📋 Model Info:")
    print(stt.get_model_info())
