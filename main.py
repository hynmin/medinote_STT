"""
STT 실행 메인 파일
"""
import argparse
from pathlib import Path
from stt_engine import MedicalSTT
from db_storage import init_db, save_transcript, save_metrics, save_summary
from metrics import compute_metrics, compute_rtf
from config import STTConfig
from summary import generate_summary
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


def load_reference_text(args):
    """
    평가용 참조 텍스트 로드

    TODO: 향후 의료 상담 평가 지표를 재정의할 때 이 함수와 관련 코드를 수정/삭제
    """
    ref_text = args.ref_text
    if not ref_text and args.ref_file:
        try:
            with open(args.ref_file, "r", encoding="utf-8") as rf:
                ref_text = rf.read()
        except Exception as e:
            print(f"⚠️ Failed to read ref file: {e}")
    return ref_text


def main():
    parser = argparse.ArgumentParser(description="의료 상담 음성을 텍스트로 변환")
    
    parser.add_argument(
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
    
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="DB 저장 비활성화 (기본: DB에 저장)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/output/transcripts.db",
        help="SQLite DB 파일 경로 (default: data/output/transcripts.db)"
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="평가용 참조 텍스트(주어지면 WER/CER 계산)"
    )
    parser.add_argument(
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
    args = parser.parse_args()

    # STT 엔진 초기화
    stt = MedicalSTT(
        model_type=args.model,
        noise_reduction=not args.no_noise_reduction
    )

    # DB 초기화 (기본적으로 활성화, --no-db로 비활성화 가능)
    save_to_db = not args.no_db
    if save_to_db:
        init_db(args.db_path)
    
    audio_path = Path(args.audio_path)
    
    # 단일 파일 처리
    if audio_path.is_file():
        result = stt.transcribe(
            str(audio_path),
            save_result=False  # JSON 파일 생성 안 함 (DB만 사용)
        )

        # 변환 결과 출력
        print("\n" + "="*50)
        print("📄 변환 결과:")
        print("="*50)
        print(result["text"])

        # DB 저장 (기본 활성화)
        if save_to_db:
            # RTF 계산
            rtf_info = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))

            tid = save_transcript(
                result,
                result.get("processing_time"),
                result.get("audio_duration"),
                rtf_info.get("rtf"),
                not args.no_noise_reduction,
                args.db_path
            )
            print(f"🗄️  Saved to DB: {args.db_path} (transcript_id={tid})")

            # RTF 출력
            if result.get("audio_duration", 0) > 0:
                print(f"\n⚡ Performance")
                print(f"  RTF: {rtf_info['rtf']:.4f} (실시간보다 {rtf_info['speed_factor']:.2f}배 빠름)")
                print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 / 오디오 길이: {result.get('audio_duration', 0):.2f}초")

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
                        medication=summary_result["medication"],
                        lifestyle_management=summary_result["lifestyle_management"],
                        model=summary_result["model"],
                        summary_time=summary_result["summary_time"],
                        db_path=args.db_path
                    )

                    # 터미널에 요약 출력
                    print("\n" + "="*50)
                    print("🤖 AI 요약")
                    print("="*50)
                    print(f"\n📌 주요 증상:")
                    print(f"  {summary_result['chief_complaint']}")
                    print(f"\n🏥 진단:")
                    print(f"  {summary_result['diagnosis']}")
                    print(f"\n💊 약물 처방:")
                    print(f"  {summary_result['medication']}")
                    print(f"\n🏃 생활 관리:")
                    for line in summary_result['lifestyle_management'].split('\n'):
                        if line.strip():
                            print(f"  - {line.strip()}")
                    print(f"\n  ↳ 요약 생성 시간: {summary_result['summary_time']}초 (summary_id={summary_id})")

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
            if save_to_db:
                # tid exists only if DB saving is enabled
                save_metrics(tid, m, args.db_path)
                print("  ↳ saved to DB (metrics)")

    else:
        print(f"❌ Invalid audio file path: {audio_path}")


if __name__ == "__main__":
    main()
