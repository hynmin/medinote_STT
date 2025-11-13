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

try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("⚠️ Warning: pyannote.audio not installed. Speaker diarization disabled.")


class MedicalSTT:
    """의료 상담 음성을 텍스트로 변환하는 STT 엔진"""

    def __init__(self, model_type="fast", enable_diarization=False, noise_reduction=False, vad_filter=False):
        """
        Args:
            model_type: "fast", "balanced", "accurate" 중 하나
            enable_diarization: 화자 분리 사용 여부
            noise_reduction: 노이즈 제거 전처리 사용 여부
            vad_filter: VAD (Voice Activity Detection) 필터 사용 여부
        """
        self.model_type = model_type
        self.model_name = STTConfig.get_model(model_type)
        self.enable_diarization = enable_diarization and DIARIZATION_AVAILABLE
        self.noise_reduction = noise_reduction
        self.vad_filter = vad_filter

        print(f"🔄 Loading {self.model_name}...")
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=STTConfig.get_device(),
            return_timestamps=True  # 타임스탬프 필수 (화자 분리용)
        )
        print(f"✅ Model loaded successfully!")

        # 화자 분리 모델 로드 (선택적)
        self.diarization_pipeline = None
        if self.enable_diarization:
            try:
                print("🔄 Loading speaker diarization model...")
                hf_token = os.getenv("HF_TOKEN")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
                print("✅ Diarization model loaded!")
            except Exception as e:
                print(f"⚠️ Diarization loading failed: {e}")
                print("   Continuing without speaker separation...")
                self.enable_diarization = False
    
    def transcribe(self, audio_path, save_result=True, reference_text=None):
        """
        오디오 파일을 텍스트로 변환 (화자 분리 옵션 포함)

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

                # 화자 분리 활성화 시 추가:
                "segments": [...],  # 화자별 세그먼트
                "num_speakers": 2,  # 화자 수

                # reference_text 제공 시 추가:
                "metrics": {"wer": 0.05, "cer": 0.02}
            }
        """
        if self.enable_diarization:
            return self._transcribe_with_diarization(audio_path, save_result, reference_text)
        else:
            return self._transcribe_simple(audio_path, save_result, reference_text)

    def _transcribe_simple(self, audio_path, save_result=True, reference_text=None):
        """화자 분리 없이 단순 변환"""
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
                "rtf": round(processing_time / max(audio_duration, 0.001), 4),
                "num_speakers": 0
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
                "rtf": round(processing_time / max(audio_duration, 0.001), 4),
                "num_speakers": 0
            }

        # STT 수행 (ffmpeg 미설치 시 librosa로 대체 로딩)
        generate_kwargs = {
            "language": STTConfig.LANGUAGE,
            "task": "transcribe"
        }

        # VAD 필터 설정
        if self.vad_filter:
            print("  🎯 Using VAD filter (Voice Activity Detection)...")

        # 노이즈 제거 전처리 (이미 로드된 오디오 사용)
        if self.noise_reduction:
            print("  🔧 Applying noise reduction...")
            y = self._apply_noise_reduction(y, sr)
            audio_input = {"array": np.asarray(y), "sampling_rate": sr}
        else:
            audio_input = {"array": np.asarray(y), "sampling_rate": sr}

        try:
            result = self.transcriber(audio_input, generate_kwargs=generate_kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "ffmpeg" in msg or "torchcodec" in msg or "libtorchcodec" in msg:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                if self.noise_reduction:
                    y = self._apply_noise_reduction(y, sr)
                audio_input = {"array": np.asarray(y), "sampling_rate": sr}
                result = self.transcriber(audio_input, generate_kwargs=generate_kwargs)
            else:
                raise

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

    def _transcribe_with_diarization(self, audio_path, save_result=True, reference_text=None):
        """화자 분리를 포함한 변환"""
        print(f"\n🎤 Processing with speaker separation: {audio_path}")
        start_time = time.time()

        # 오디오 길이 측정
        audio_duration = self._get_audio_duration(audio_path)

        # 1단계: 화자 분리 (누가 언제 말했는지)
        speaker_segments = []
        if self.diarization_pipeline:
            print("  📊 Step 1/2: Detecting speakers...")
            try:
                diarization = self.diarization_pipeline(audio_path)
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end
                    })
                print(f"  ✅ Found {len(set(s['speaker'] for s in speaker_segments))} speakers")
            except Exception as e:
                msg = str(e).lower()
                if "ffmpeg" in msg:
                    print("  ⚠️ FFmpeg not found. Skipping diarization.")
                else:
                    print(f"  ⚠️ Diarization failed: {e}")
                speaker_segments = [{
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 999999.0
                }]
        else:
            speaker_segments = [{
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 999999.0
            }]

        # 2단계: 음성 인식 (뭐라고 말했는지)
        print("  🎯 Step 2/2: Transcribing speech...")
        generate_kwargs = {
            "language": STTConfig.LANGUAGE,
            "task": "transcribe"
        }

        # VAD 필터 설정
        if self.vad_filter:
            generate_kwargs["vad_filter"] = True
            generate_kwargs["chunk_length_s"] = 30
            print("  🎯 Using VAD filter...")

        # 노이즈 제거 전처리
        if self.noise_reduction:
            audio_input = self._preprocess_audio(audio_path)
        else:
            audio_input = audio_path

        try:
            transcription = self.transcriber(audio_input, generate_kwargs=generate_kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "ffmpeg" in msg or "torchcodec" in msg or "libtorchcodec" in msg:
                y, sr = librosa.load(audio_path, sr=16000, mono=True)
                if self.noise_reduction:
                    y = self._apply_noise_reduction(y, sr)
                audio_input = {"array": np.asarray(y), "sampling_rate": sr}
                transcription = self.transcriber(audio_input, generate_kwargs=generate_kwargs)
            else:
                raise

        # 3단계: 화자와 텍스트 매칭
        result_segments = []

        # Whisper의 chunks (타임스탬프 포함 텍스트)
        if "chunks" in transcription:
            chunks = transcription["chunks"]
        else:
            chunks = [{
                "text": transcription["text"],
                "timestamp": (0.0, None)
            }]

        for chunk in chunks:
            text = chunk["text"].strip()
            if not text:
                continue

            chunk_start = chunk["timestamp"][0] if chunk["timestamp"][0] else 0.0
            chunk_end = chunk["timestamp"][1] if chunk["timestamp"][1] else chunk_start + 1.0

            # 이 텍스트를 말한 화자 찾기
            speaker = self._find_speaker(chunk_start, chunk_end, speaker_segments)

            result_segments.append({
                "speaker": speaker,
                "text": text,
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2)
            })

        processing_time = time.time() - start_time

        # 결과 정리
        output = {
            "text": transcription["text"],
            "segments": result_segments,
            "num_speakers": len(set(s["speaker"] for s in result_segments)),
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

        print(f"✅ Transcription complete!")
        print(f"   💬 {len(result_segments)} segments from {output['num_speakers']} speakers")

        # 결과 저장
        if save_result:
            self._save_result(output)

        return output

    def _find_speaker(self, chunk_start, chunk_end, speaker_segments):
        """텍스트 구간에 해당하는 화자 찾기"""
        chunk_mid = (chunk_start + chunk_end) / 2

        for segment in speaker_segments:
            if segment["start"] <= chunk_mid <= segment["end"]:
                return segment["speaker"]

        # 못 찾으면 가장 가까운 화자
        if speaker_segments:
            return speaker_segments[0]["speaker"]

        return "SPEAKER_00"

    def _get_audio_duration(self, audio_path):
        """오디오 파일 길이(초) 측정"""
        try:
            import soundfile as sf
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except Exception:
            # soundfile 실패 시 librosa로 대체
            try:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            except Exception:
                return 0.0

    def _preprocess_audio(self, audio_path):
        """오디오 전처리 (노이즈 제거)"""
        print("  🔧 Applying noise reduction...")
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        y = self._apply_noise_reduction(y, sr)
        return {"array": np.asarray(y), "sampling_rate": sr}

    def _apply_noise_reduction(self, audio_array, sample_rate):
        """노이즈 제거 적용"""
        try:
            import noisereduce as nr
            # 노이즈 프로파일 자동 추정 및 제거
            reduced = nr.reduce_noise(
                y=audio_array,
                sr=sample_rate,
                stationary=True,  # 배경소음이 일정한 경우
                prop_decrease=1.0  # 노이즈 감소 정도 (0.0~1.0)
            )
            return reduced
        except ImportError:
            print("  ⚠️ noisereduce not installed. Skipping noise reduction.")
            return audio_array
        except Exception as e:
            print(f"  ⚠️ Noise reduction failed: {e}")
            return audio_array
    
    def transcribe_batch(self, audio_paths, save_result=True):
        """
        여러 오디오 파일을 일괄 변환
        
        Args:
            audio_paths: 오디오 파일 경로 리스트
            save_result: 결과를 저장할지 여부
            
        Returns:
            list: 변환 결과 리스트
        """
        results = []
        
        print(f"\n📦 Batch processing {len(audio_paths)} files...")
        
        for i, audio_path in enumerate(audio_paths, 1):
            print(f"\n[{i}/{len(audio_paths)}]", end=" ")
            result = self.transcribe(audio_path, save_result=False)
            results.append(result)
        
        # 일괄 저장
        if save_result:
            self._save_batch_result(results)
        
        return results
    
    def _save_result(self, result):
        """단일 결과를 JSON 파일로 저장"""
        audio_name = Path(result["audio_file"]).stem
        suffix = "_with_speakers" if "segments" in result else "_transcription"
        output_path = Path(STTConfig.OUTPUT_DIR) / f"{audio_name}{suffix}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved to: {output_path}")
    
    def _save_batch_result(self, results):
        """배치 결과를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(STTConfig.OUTPUT_DIR) / f"batch_transcription_{timestamp}.json"
        
        batch_summary = {
            "total_files": len(results),
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Batch results saved to: {output_path}")
    
    def print_conversation(self, result):
        """대화 형식으로 보기 좋게 출력 (화자 분리 결과용)"""
        if "segments" not in result:
            print("\n⚠️ No speaker segments found (diarization was not enabled)")
            return

        print("\n" + "="*60)
        print("📋 대화 내용")
        print("="*60)

        for segment in result["segments"]:
            speaker = segment["speaker"]
            text = segment["text"]
            time_info = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"

            print(f"\n{speaker} {time_info}")
            print(f"  {text}")

        print("\n" + "="*60)

    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": "GPU" if STTConfig.get_device() == 0 else "CPU",
            "language": STTConfig.LANGUAGE,
            "diarization_enabled": self.enable_diarization
        }


# 간단한 사용 예시
if __name__ == "__main__":
    # 화자 분리 없이 사용
    stt = MedicalSTT(model_type="fast")
    result = stt.transcribe("audio.mp3")

    # 화자 분리 포함
    # stt = MedicalSTT(model_type="fast", enable_diarization=True)
    # result = stt.transcribe("audio.mp3")
    # stt.print_conversation(result)

    print("\n📋 Model Info:")
    print(stt.get_model_info())
