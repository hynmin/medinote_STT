"""
STT 실행 메인 파일
"""
import argparse
from pathlib import Path
from stt_engine import MedicalSTT
from storage import init_db, save_transcript, save_segments, save_metrics
from metrics import compute_metrics
from config import STTConfig


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
        "--diarization",
        action="store_true",
        help="화자 분리 사용 (의사/환자 구별)"
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
    parser.add_argument(
        "--no-vad-filter",
        action="store_true",
        help="VAD 필터 비활성화 (기본: 활성화)"
    )

    args = parser.parse_args()

    # STT 엔진 초기화 (기본적으로 noise_reduction과 vad_filter 활성화)
    stt = MedicalSTT(
        model_type=args.model,
        enable_diarization=args.diarization,
        noise_reduction=not args.no_noise_reduction,
        vad_filter=not args.no_vad_filter
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

        # 화자 분리 사용시
        if args.diarization:
            stt.print_conversation(result)
            print(f"\n📊 요약")
            print(f"  화자 수: {result['num_speakers']}명")
            print(f"  대화 구간: {len(result['segments'])}개")
        else:
            # 화자 분리 없으면 텍스트만 출력
            print("\n" + "="*50)
            print("📄 변환 결과:")
            print("="*50)
            print(result["text"])

        # DB 저장 (기본 활성화)
        if save_to_db:
            # RTF 계산
            from metrics import compute_rtf
            rtf_info = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))

            tid = save_transcript(
                result,
                result.get("processing_time"),
                result.get("audio_duration"),
                rtf_info.get("rtf"),
                not args.no_noise_reduction,
                not args.no_vad_filter,
                args.db_path
            )
            save_segments(tid, result.get("segments", []), args.db_path)
            print(f"🗄️  Saved to DB: {args.db_path} (transcript_id={tid})")

        # 평가지표 계산/출력/저장 (옵션)
        ref_text = args.ref_text
        if not ref_text and args.ref_file:
            try:
                with open(args.ref_file, "r", encoding="utf-8") as rf:
                    ref_text = rf.read()
            except Exception as e:
                print(f"⚠️ Failed to read ref file: {e}")
        if ref_text:
            m = compute_metrics(ref_text, result.get("text", ""))
            print("\n📐 Metrics")
            print(f"  WER: {m['wer']:.4f}  CER: {m['cer']:.4f}")
            if save_to_db:
                # tid exists only if DB saving is enabled
                save_metrics(tid, m, args.db_path)
                print("  ↳ saved to DB (metrics)")

        # RTF 계산 및 출력
        if result.get("audio_duration", 0) > 0:
            from metrics import compute_rtf
            rtf_info = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))
            print(f"\n⚡ Performance")
            print(f"  RTF: {rtf_info['rtf']:.4f} (실시간보다 {rtf_info['speed_factor']:.2f}배 빠름)")
            print(f"  처리 시간: {result.get('processing_time', 0):.2f}초 / 오디오 길이: {result.get('audio_duration', 0):.2f}초")
    
    # 디렉토리 내 모든 오디오 파일 처리
    elif audio_path.is_dir():
        audio_files = list(audio_path.glob("*.mp3")) + \
                     list(audio_path.glob("*.wav")) + \
                     list(audio_path.glob("*.m4a"))
        
        if not audio_files:
            print(f"❌ No audio files found in {audio_path}")
            return
        
        print(f"\n📦 Processing {len(audio_files)} files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
            result = stt.transcribe(
                str(audio_file),
                save_result=False  # JSON 파일 생성 안 함 (DB만 사용)
            )
            if save_to_db:
                # RTF 계산
                from metrics import compute_rtf
                rtf_info = compute_rtf(result.get("processing_time", 0), result.get("audio_duration", 0))

                tid = save_transcript(
                    result,
                    result.get("processing_time"),
                    result.get("audio_duration"),
                    rtf_info.get("rtf"),
                    not args.no_noise_reduction,
                    not args.no_vad_filter,
                    args.db_path
                )
                save_segments(tid, result.get("segments", []), args.db_path)
                print(f"🗄️  Saved to DB: {args.db_path} (transcript_id={tid})")

            # 파일별 평가지표(참조가 제공된 경우)
            ref_text = args.ref_text
            if not ref_text and args.ref_file:
                try:
                    with open(args.ref_file, "r", encoding="utf-8") as rf:
                        ref_text = rf.read()
                except Exception as e:
                    print(f"⚠️ Failed to read ref file: {e}")
            if ref_text:
                m = compute_metrics(ref_text, result.get("text", ""))
                print(f"  📐 WER: {m['wer']:.4f}  CER: {m['cer']:.4f}")
                if save_to_db:
                    save_metrics(tid, m, args.db_path)
        
        print("\n" + "="*50)
        print(f"📊 배치 처리 완료: {len(audio_files)}개 파일")
        print("="*50)
    
    else:
        print(f"❌ Invalid path: {audio_path}")


if __name__ == "__main__":
    main()
