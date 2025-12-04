"""
개발 평가지표 계산 유틸리티 (WER/CER/RTF)
"""
from jiwer import wer, cer


def _normalize_number(text: str) -> str:
    """숫자를 한글로 변환 (간단한 버전)"""
    import re

    # 숫자 -> 한글 매핑
    num_map = {
        '0': '영', '1': '일', '2': '이', '3': '삼', '4': '사',
        '5': '오', '6': '육', '7': '칠', '8': '팔', '9': '구'
    }

    def replace_number(match):
        num = match.group(0)
        # 간단하게 각 자릿수를 한글로 변환 (예: 123 -> 일이삼)
        return ''.join(num_map.get(d, d) for d in num)

    # 연속된 숫자를 찾아서 한글로 변환
    return re.sub(r'\d+', replace_number, text)


def compute_metrics(ref_text: str, hyp_text: str, remove_fillers=True) -> dict:
    """참조(ref)와 가설(hyp) 텍스트로 WER/CER 계산.

    Args:
        ref_text: 참조 텍스트
        hyp_text: 가설 텍스트
        remove_fillers: 추임새 및 정규화 적용 여부 (기본값: True)

    Returns: { 'wer': float, 'cer': float, 'ref_chars': int, 'hyp_chars': int }
    """
    import re
    ref = (ref_text or "").strip()
    hyp = (hyp_text or "").strip()

    # 줄바꿈(\n, \r\n)을 공백으로 치환
    ref = ref.replace('\n', ' ').replace('\r', ' ')
    hyp = hyp.replace('\n', ' ').replace('\r', ' ')

    # 연속된 공백을 하나로 축약
    ref = re.sub(r'\s+', ' ', ref).strip()
    hyp = re.sub(r'\s+', ' ', hyp).strip()

    # 정규화 (선택적)
    if remove_fillers:
        # 1. 영어 소문자 변환
        ref = ref.lower()
        hyp = hyp.lower()

        # 2. 숫자를 한글로 변환
        ref = _normalize_number(ref)
        hyp = _normalize_number(hyp)

        # 3. 구두점 제거
        punctuation = r'[.,!?;:\'\"\'\'\"\"\(\)\[\]\{\}<>…·]'
        ref = re.sub(punctuation, '', ref)
        hyp = re.sub(punctuation, '', hyp)

        # 4. 한국어 추임새/감탄사 제거
        fillers = [
            r'\b음+\b',           # 음, 음음, 음음음
            r'\b으+[음응]\b',     # 으음, 으응, 으으음
            r'\b응+\b',           # 응, 응응, 응응응
            r'\b어+\b',           # 어, 어어
            r'\b아+\b',           # 아, 아아
            r'\b에+\b',           # 에, 에에
            r'\b그+\b',           # 그 (단독으로 쓰일 때)
            r'\b이+\b',           # 이 (단독으로 쓰일 때)
            r'\b네+\b',           # 네, 네네
            r'\b예+\b',           # 예, 예예
            r'\b뭐+\b',           # 뭐, 뭐뭐
            r'\b저+\b',           # 저 (단독으로 쓰일 때)
            r'\b자+\b',           # 자 (단독으로 쓰일 때)
            r'\b좀+\b',           # 좀
            r'\b막+\b',           # 막
            r'\b약간+\b',         # 약간
            r'\b진짜+\b',         # 진짜
            r'\b되게+\b',         # 되게
        ]

        for filler in fillers:
            ref = re.sub(filler, '', ref)
            hyp = re.sub(filler, '', hyp)

        # 5. 모든 공백 제거 후 다시 단어별로 분리 (띄어쓰기 영향 최소화)
        # 한글, 영문, 숫자를 기준으로 토큰화
        ref_tokens = re.findall(r'[가-힣a-z]+', ref)
        hyp_tokens = re.findall(r'[가-힣a-z]+', hyp)

        # 토큰을 공백으로 연결 (띄어쓰기 통일)
        ref = ' '.join(ref_tokens)
        hyp = ' '.join(hyp_tokens)

    return {
        "wer": float(wer(ref, hyp)),
        "cer": float(cer(ref, hyp)),
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
    }


def compute_rtf(processing_time: float, audio_length: float) -> float:
    """RTF (Real-time Factor) 계산.

    Args:
        processing_time: 처리 시간(초)
        audio_length: 오디오 길이(초)

    Returns: rtf (float)
    """
    if audio_length is None or audio_length <= 0:
        return 0.0

    return round(processing_time / audio_length, 4)
