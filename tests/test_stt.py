"""
STT 엔진 테스트
"""
import unittest
from pathlib import Path
import sys

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from stt_engine import MedicalSTT
from config import STTConfig


class TestSTTEngine(unittest.TestCase):
    """STT 엔진 테스트"""
    
    def setUp(self):
        """테스트 초기화"""
        self.stt = MedicalSTT(model_type="fast")
    
    def test_model_loading(self):
        """모델 로딩 테스트"""
        self.assertIsNotNone(self.stt.transcriber)
        self.assertEqual(self.stt.model_type, "fast")
    
    def test_config(self):
        """설정 테스트"""
        model_name = STTConfig.get_model("fast")
        self.assertEqual(model_name, "openai/whisper-small")
        
        model_name = STTConfig.get_model("balanced")
        self.assertEqual(model_name, "openai/whisper-medium")
    
    def test_get_model_info(self):
        """모델 정보 조회 테스트"""
        info = self.stt.get_model_info()
        self.assertIn("model_name", info)
        self.assertIn("device", info)
        self.assertEqual(info["language"], "korean")
    
    # 실제 오디오 파일이 있을 때만 동작
    def test_transcribe(self):
        """음성 변환 테스트 (오디오 파일 필요)"""
        test_audio = Path(STTConfig.AUDIO_DIR) / "test.mp3"
        
        if test_audio.exists():
            result = self.stt.transcribe(str(test_audio), save_result=False)
            self.assertIn("text", result)
            self.assertIn("processing_time", result)
        else:
            self.skipTest(f"Test audio file not found: {test_audio}")


if __name__ == "__main__":
    unittest.main()
