"""
OpenAI를 사용한 의료 상담 요약 모듈
"""
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()


def generate_summary(transcript_text: str, model: str = "gpt-4o-mini") -> dict:
    """
    STT 결과를 OpenAI로 요약

    Args:
        transcript_text: STT로 변환된 전체 대화 텍스트
        model: 사용할 OpenAI 모델 (기본: gpt-4o-mini)

    Returns:
        dict: {
            "chief_complaint": "증상",
            "diagnosis": "진단",
            "recommendation": "권고사항",
            "model": "gpt-4o-mini",
            "summary_time": 1.23  # 초
        }
    """
    # API 키 가져오기
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다")

    # OpenAI 클라이언트 생성
    client = OpenAI(api_key=api_key)

    # 프롬프트 작성
    prompt = f"""
당신은 의료 상담 기록을 분석하는 전문가입니다.
다음 의료 상담 대화를 읽고 어린아이도 이해할 수 있게 쉬운 말로 요약 정리해주세요:

[대화 내용]
{transcript_text}

다음 형식으로 정리해주세요. 각 섹션은 명확하게 구분하고, 해당 내용이 없으면 "없음" 또는 "해당 없음"으로 작성하세요:

1. 증상:
(환자의 증상을 간결하게 정리)

2. 진단:
(의사의 진단명 또는 추정 진단 간결하게 정리)

3. 권고사항:
(처방된 약물명, 용량, 복용 방법. 식이요법, 운동, 권고사항, 주의사항, 재방문 일정 등)
"""

    # API 호출 시간 측정
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 의료 상담 기록을 이해하기 쉬운 말로 정확하게 요약하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 낮은 값 = 더 일관되고 정확한 응답
            max_tokens=1000   # 최대 토큰 수
        )

        summary_time = time.time() - start_time

        # GPT 응답 파싱
        content = response.choices[0].message.content.strip()

        # 응답을 3개 섹션으로 분리
        sections = parse_summary_sections(content)

        return {
            "chief_complaint": sections.get("증상", ""),
            "diagnosis": sections.get("진단", ""),
            "recommendation": sections.get("권고사항", ""),
            "model": model,
            "summary_time": round(summary_time, 2)
        }

    except Exception as e:
        print(f"❌ OpenAI API 호출 실패: {e}")
        return {
            "chief_complaint": "요약 생성 실패",
            "diagnosis": "요약 생성 실패",
            "recommendation": "요약 생성 실패",
            "model": model,
            "summary_time": 0.0
        }


def parse_summary_sections(content: str) -> dict:
    """
    응답을 3개 섹션으로 파싱
    """
    sections = {}
    current_section = None
    lines = []

    for line in content.split('\n'):
        line = line.strip()

        # 섹션 제목 감지 (1. 증상:, 2. 진단: 등)
        if line.startswith("1.") and "증상" in line:
            if current_section and lines:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "증상"
            lines = []
        elif line.startswith("2.") and "진단" in line:
            if current_section and lines:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "진단"
            lines = []
        elif line.startswith("3.") and "권고사항" in line:
            if current_section and lines:
                sections[current_section] = '\n'.join(lines).strip()
            current_section = "권고사항"
            lines = []
        elif current_section and line:
            # 섹션 내용 수집
            lines.append(line)

    # 마지막 섹션 저장
    if current_section and lines:
        sections[current_section] = '\n'.join(lines).strip()

    return sections
