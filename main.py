"""
STT 실행 메인 파일
"""
import argparse
from pathlib import Path
from models.stt.engine.whisper_engine import MedicalSTT, OpenAIWhisperSTT
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

    TODO: 향후 의료 상담 평가 지표를 재정의할 때 이 함수와 관련 코드를 수정/삭제
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
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["fast", "balanced", "accurate"],
        default="fast",
        help="사용할 모델 (default: fast)"
    )

    parser.add_argument( #개발단계 wer/cer 계산용
        "--ref-file",
        type=str,
        default=None,
        help="평가용 참조 텍스트 파일 경로(UTF-8)"
    )
    
    parser.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="노이즈 제거 비활성화 (기본: 활성화)"
    )
    
    parser.add_argument(
        "--vad",
        action="store_true",
        help="VAD(Voice Activity Detection) 사용 - 무음 구간 제거"
    )

    parser.add_argument(
        "--use-openai-api",
        type=str,
        nargs="?",
        const="whisper-1",
        default=None,
        choices=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
        help="OpenAI API 사용. 모델 선택: whisper-1(기본), gpt-4o-transcribe, gpt-4o-mini-transcribe"
    )
    args = parser.parse_args()

    # STT 엔진 초기화
    if args.use_openai_api:
        # OpenAI API 사용
        stt = OpenAIWhisperSTT(model=args.use_openai_api)
        print(f"🌐 Using OpenAI API: {args.use_openai_api}")
    else:
        # 로컬 Hugging Face 모델 사용
        stt = MedicalSTT(
            model_type=args.model,
            noise_reduction=not args.no_noise_reduction,
            use_vad=args.vad
        )

    # DB 초기화
    db_path = STTConfig.DB_PATH
    init_db(db_path)
    
    audio_path = Path(args.audio_path)
    
    # 단일 파일 처리
    if audio_path.is_file():
        result = stt.transcribe(
            str(audio_path),
        )

        # 변환 결과 출력
        print("\n" + "="*50)
        print("📄 변환 결과:")
        print("="*50)
        print(result["text"])

        # DB 저장 구분 필요한지 확인 필요
        if True:
            # RTF 계산
            rtf_info = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))

            tid = save_transcript(
                result, # STT 결과 dict (audio_file, model, text 포함)
                result.get("processing_time"),
                result.get("audio_duration"),
                rtf_info.get("rtf"),
                not args.no_noise_reduction,
                db_path
            )
            print(f"🗄️  Saved to DB: {db_path} (transcript_id={tid})")

            # RTF 출력
            audio_duration = result.get("audio_duration")
            if audio_duration and audio_duration > 0:
                print(f"\n⚡ Performance")
                rtf_value = rtf_info['rtf']
                if rtf_value <= 1.0:
                    print(f"  RTF: {rtf_value:.4f} (실시간보다 {1/rtf_value:.2f}배 빠름)")
                else:
                    print(f"  RTF: {rtf_value:.4f} (실시간보다 {rtf_value:.2f}배 느림)")
                print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 / 오디오 길이: {audio_duration:.2f}초")
            else:
                print(f"\n⚡ Performance")
                print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 (RTF 계산 불가 - 오디오 길이 정보 없음)")

            # AI 요약 생성 (텍스트가 있을 때만)
            if result["text"].strip():
                print("\n🤖 AI 요약 생성 중...")
                try:
                    summary_result = generate_summary(
                        transcript_text=result["text"],
                        model="gpt-4o-mini"
                    )

                    summary_id = save_summary(
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

        # 평가지표 계산/출력/저장 (옵션)
        # TODO: 향후 의료 상담 평가 지표를 재정의할 때 이 블록을 수정/삭제
        ref_text = load_reference_text(args)
        if ref_text:
            m = compute_metrics(ref_text, result.get("text", ""))
            print("\n📐 Metrics")
            print(f"  WER: {m['wer']:.4f}  CER: {m['cer']:.4f}")

    else:
        print(f"❌ Invalid audio file path: {audio_path}")


if __name__ == "__main__":
    main()
