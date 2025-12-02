"""
STT 실행 메인 파일
"""
import argparse
from pathlib import Path
from models.stt.engine.hf_engine import HFWhisperSTT
from models.stt.engine.openai_engine import OpenAIWhisperSTT
from db.storage import init_db, save_transcript, save_summary
from models.stt.utils.metrics import compute_metrics, compute_rtf
from models.stt.core.config import STTConfig
from models.stt.pipelines.summarize import generate_summary
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def load_reference_text(args):
    """
    평가용 참조 텍스트 로드
    """
    ref_text = None
    if args.ref_file:
        try:
            with open(args.ref_file, "r", encoding="utf-8") as rf:
                ref_text = rf.read()
        except Exception as e:
            print(f"⚠️ Failed to read ref file: {e}")
    return ref_text


def main():
    parser = argparse.ArgumentParser(description="의료 상담 음성을 텍스트로 변환")

    parser.add_argument( #cli 테스트용. data/audio/파일
        "audio_path",
        type=str,
        help="오디오 파일 경로 또는 디렉토리"
    )  
    parser.add_argument( #모델 선택
        "--model",
        type=str,
        choices=STTConfig.MODEL_CHOICES,
        default=STTConfig.DEFAULT_MODEL,
        help="사용할 모델 (default: fast)"
    )
    parser.add_argument( #개발단계 wer/cer 계산용
        "--ref-file",
        type=str,
        default=None,
        help="평가용 참조 텍스트 파일 경로(UTF-8)"
    )
    parser.add_argument( #노이즈 제거
        "--no-noise-reduction",
        action="store_true",
        help="노이즈 제거 비활성화 (기본: 활성화, HF 모델만 적용)"
    )
    parser.add_argument( #음성활동감지
        "--vad",
        action="store_true",
        help="VAD(Voice Activity Detection) 사용 (HF 모델만 적용)"
    )

    args = parser.parse_args()

    # STT 엔진 선택
    if STTConfig.is_api_model(args.model):
        # OpenAI API 모델
        stt = OpenAIWhisperSTT(model=args.model)
        print(f"Using OpenAI API: {args.model}")
    else:
        # HuggingFace 로컬 모델
        stt = HFWhisperSTT(
            model=args.model,
            noise_reduction=not args.no_noise_reduction,
            use_vad=args.vad
        )

    # 테이블 없으면 생성
    db_path = STTConfig.DB_PATH
    init_db(db_path)
    
    audio_path = Path(args.audio_path)
    
    # 음성파일 STT 처리
    if audio_path.is_file(): 
        result = stt.transcribe(   # 음성파일 STT 처리
            str(audio_path),
        )

        # 변환 결과 출력
        print("\n" + "="*50)
        print("📄 변환 결과:")
        print("="*50)
        print(result["text"])

        # RTF 계산 및 cli 출력
        rtf = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))
        audio_duration = result.get("audio_duration")
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)  # bytes to MB

        print(f"\n⚡ Performance")
        print(f"  파일 크기: {file_size_mb:.2f} MB")
        if audio_duration and audio_duration > 0:
            if rtf <= 1.0:
                print(f"  RTF: {rtf:.4f} (실시간보다 {1/rtf:.2f}배 빠름)")
            else:
                print(f"  RTF: {rtf:.4f} (실시간보다 {rtf:.2f}배 느림)")
            print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 / 오디오 길이: {audio_duration:.2f}초")
        else:
            print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 (RTF 계산 불가 - 오디오 길이 정보 없음)")

        # STT 결과 DB저장
        tid = save_transcript(
            result,
            result.get("processing_time"),
            result.get("audio_duration"),
            rtf,
            not args.no_noise_reduction,
            db_path
        )
        print(f"🗄️  Saved to DB: {db_path} (transcript_id={tid})")

        # AI 요약 생성
        if result["text"].strip():
            print("\n🤖 AI 요약 생성 중...")
            try:
                summary_result = generate_summary(  # 요약정리 생성
                    transcript_text=result["text"],
                    model="gpt-4o-mini"
                )

                summary_id = save_summary(          # 요약정리 DB 저장
                    transcript_id=tid,
                    chief_complaint=summary_result["chief_complaint"],
                    diagnosis=summary_result["diagnosis"],
                    recommendation=summary_result["recommendation"],
                    model=summary_result["model"],
                    summary_time=summary_result["summary_time"],
                    db_path=db_path
                )

                # 터미널에 요약 출력
                print("\n" + "="*50)
                print("AI 요약")
                print("="*50)
                print(f"\n  증상:")
                print(f"  {summary_result['chief_complaint']}")
                print(f"\n  진단:")
                print(f"  {summary_result['diagnosis']}")
                print(f"\n 소견:")
                for line in summary_result['recommendation'].split('\n'):
                    if line.strip():
                        print(line)
                print(f"\n 요약 생성 시간: {summary_result['summary_time']}초 (summary_id={summary_id})")

            except Exception as e:
                print(f"⚠️  AI 요약 생성 실패: {e}")
        else:
            print("\n⏭️  텍스트가 비어있어 AI 요약을 건너뜁니다.")

        # 평가지표 cli 출력 (개발단계)
        ref_text = load_reference_text(args)
        if ref_text:
            m = compute_metrics(ref_text, result.get("text", ""))
            print("\n📐 Metrics")
            print(f"  WER: {m['wer']:.4f}  CER: {m['cer']:.4f}")
            print(f"  참조 글자수: {m['ref_chars']}  인식 글자수: {m['hyp_chars']}")

    else:
        print(f"❌ Invalid audio file path: {audio_path}")


if __name__ == "__main__":
    main()
