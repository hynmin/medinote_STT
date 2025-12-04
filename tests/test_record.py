"""
마이크 녹음 모듈 (로컬 테스트용)
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

# 녹음 설정
SAMPLE_RATE = 16000  # Whisper 권장 샘플링 레이트
CHANNELS = 1  # 모노
RECORDINGS_DIR = Path("tests/test_recordings")  # CLI 녹음 테스트용

# 디렉토리 생성
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


class AudioRecorder:
    """간단한 오디오 녹음기"""

    def __init__(self):
        self.recording = False
        self.frames = []

    def start(self):
        """녹음 시작"""
        self.recording = True
        self.frames = []
        print("\n🔴 녹음 중... (Space 키로 중지)")

        def callback(indata, frames, time, status):
            if status:
                print(f"⚠️  {status}")
            if self.recording:
                self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=callback
        )
        self.stream.start()

    def stop(self):
        """녹음 중지"""
        if not self.recording:
            return None

        self.recording = False
        self.stream.stop()
        self.stream.close()

        if not self.frames:
            print("⚠️  녹음된 데이터가 없습니다.")
            return None

        # 프레임 결합
        audio_data = np.concatenate(self.frames, axis=0)

        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RECORDINGS_DIR / f"recording_{timestamp}.wav"

        sf.write(filename, audio_data, SAMPLE_RATE)
        duration = len(audio_data) / SAMPLE_RATE

        print(f"⏹️  녹음 완료!")
        print(f"   파일: {filename}")
        print(f"   길이: {duration:.1f}초")

        return filename


def main():
    """메인 실행 함수"""
    print("="*50)
    print("🎙️  의료 상담 녹음 시스템 (로컬 테스트)")
    print("="*50)
    print("\n📝 사용법:")
    print("  [Space] 녹음 시작/중지")
    print("  [Enter] STT 처리")
    print("  [q] 종료")
    print()

    recorder = AudioRecorder()
    recorded_file = None

    # Windows에서 키 입력 받기
    try:
        import msvcrt

        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()

                # Space 키 (녹음 시작/중지)
                if key == b' ':
                    if not recorder.recording:
                        recorder.start()
                    else:
                        recorded_file = recorder.stop()

                # Enter 키 (STT 처리)
                elif key == b'\r':
                    if recorded_file and recorded_file.exists():
                        print("\n🤖 STT 처리 중...")
                        print("-" * 50)

                        # test_cli.py 실행
                        result = subprocess.run(
                            [sys.executable, "tests/test_cli.py", str(recorded_file)],
                            capture_output=False
                        )

                        print("-" * 50)
                        if result.returncode == 0:
                            print("✅ 처리 완료!\n")
                        else:
                            print("❌ 처리 실패\n")

                        recorded_file = None
                    else:
                        print("⚠️  먼저 녹음을 완료해주세요.")

                # q 키 (종료)
                elif key == b'q':
                    if recorder.recording:
                        recorder.stop()
                    print("\n👋 프로그램을 종료합니다.")
                    break

            time.sleep(0.1)

    except ImportError:
        # Linux/Mac
        print("⚠️  이 기능은 Windows에서만 작동합니다.")
        print("   Linux/Mac에서는 다음 명령어를 사용하세요:")
        print("   python -c \"import sounddevice as sd; import soundfile as sf; ...\"")


if __name__ == "__main__":
    main()
