"""
화자 분리 STT 사용 예시
"""
from stt_engine import MedicalSTT

# 1. 기본 사용 (화자 분리 포함)
print("=== 예시 1: 화자 분리 포함 ===")
stt = MedicalSTT(model_type="fast", enable_diarization=True)

# 오디오 파일 변환
result = stt.transcribe("data/audio/consultation.mp3")

# 결과 확인
print(f"\n화자 수: {result['num_speakers']}명")
print(f"대화 구간: {len(result['segments'])}개")

# 대화 내용 출력
stt.print_conversation(result)


# 2. 빠른 변환 (화자 분리 없음)
print("\n\n=== 예시 2: 화자 분리 없음 (빠름) ===")
stt_fast = MedicalSTT(model_type="fast", enable_diarization=False)
result_fast = stt_fast.transcribe("data/audio/consultation.mp3")

print(f"\n변환된 텍스트: {result_fast['text']}")


# 3. 결과 데이터 활용
print("\n\n=== 예시 3: 결과 데이터 활용 ===")

# 각 화자별로 말한 내용 분리
speaker_texts = {}
for segment in result["segments"]:
    speaker = segment["speaker"]
    text = segment["text"]
    
    if speaker not in speaker_texts:
        speaker_texts[speaker] = []
    
    speaker_texts[speaker].append(text)

# 화자별 출력
for speaker, texts in speaker_texts.items():
    print(f"\n{speaker}가 말한 내용:")
    for text in texts:
        print(f"  - {text}")


# 4. 특정 화자만 추출
print("\n\n=== 예시 4: 첫 번째 화자만 추출 ===")
first_speaker = result["segments"][0]["speaker"]
first_speaker_texts = [
    segment["text"] 
    for segment in result["segments"] 
    if segment["speaker"] == first_speaker
]

print(f"{first_speaker}: {' '.join(first_speaker_texts)}")