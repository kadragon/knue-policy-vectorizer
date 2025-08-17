import pytest

from src.knue_board_ingestor import KnueBoardIngestor


@pytest.mark.unit
def test_strip_html_cleans_lists_and_tags_and_preserves_text_tokens():
    raw = (
        '&lt;ol style="color:#666;font-size:18px">\n'
        '  &lt;li style="font-size:0.98em">&lt;p>&lt;행정예고&gt;&lt;/p>&lt;/li>\n'
        "  &lt;li>&lt;p>다음 항목&lt;/p>&lt;/li>\n"
        "&lt;/ol>"
    )
    ing = KnueBoardIngestor()
    cleaned = ing._strip_html(raw)

    # Should be plain text without HTML angle brackets/tags
    assert "<ol" not in cleaned and "</li>" not in cleaned
    assert "style=" not in cleaned
    # Should include the meaningful text from items, ideally one per line with bullets
    assert "행정예고" in cleaned
    assert "다음 항목" in cleaned
    # No raw entities should remain
    assert "&lt;" not in cleaned and "&gt;" not in cleaned


@pytest.mark.unit
def test_strip_html_removes_olli_artifacts():
    raw = (
        'olli1. 관련li  가. 제4대학-1974(2025.04.07.) "골프연습장 사용 협조 요청" li\n'
        "olli1. 관련: 개인정보보호법 제25조 ... li2. 항목 내용"
    )
    ing = KnueBoardIngestor()
    cleaned = ing._strip_html(raw)
    assert "olli" not in cleaned.lower()
    # Should preserve numbering and text
    assert "1. 관련" in cleaned or "관련:" in cleaned


@pytest.mark.unit
def test_strip_html_removes_residual_tag_tokens_and_icons():
    raw = (
        "1. 관련li  가. 제4대학-1974(2025.04.07.) “골프연습장 사용 협조 요청”  나. 2025학년도 주요업무 추진계획 보고 “총장 지시”  다. 제4대학-2815(2025.05.21.) “2025학년도 제1차 스포츠시설운영위원회 회의결과 보고” "
        "2. 2025학년도 제1차 스포츠시설운영위원회 회의 결과에 따라 제4대학에서 관리·운영하고 있는 스포츠시설 이용에 대한 변경 사항을 다음과 같이 알려드리오니, 각 부서 및 대학(원) 등에서는 소속 교직원 및 학생에게 안내하여 주시기 바랍니다. li  □ 변경 사항 tbodytr strong대상 strong변경 내용 strong시행 시기 strong비고 (변경 사유) tr  골프장 외부필드   수업 목적 외 일체 사용을 금지 2025.6.1.부터 인근주민 민원, 안전사고 위협 tr  제4대학에서 관리·운영하는 학내 모든 스포츠시설(골프연습장, 테니스장, 풋살장, 종합구장, 대운동장) 주말·휴일 사용 임시적 폐쇄 ※ 3개월간 폐쇄 후 지속 여부 재검토 2025.9.1. ~2025.11.30. 주말·휴일 기간 중 관리·통제가 사실상 불가   li"
    )
    ing = KnueBoardIngestor()
    cleaned = ing._strip_html(raw)
    # Residual tag tokens removed
    for token in [" li ", "tbody", "thead", "tr", "td", "table", "strong"]:
        assert token not in cleaned
    # Icon bullets normalized
    assert "" not in cleaned and "□" not in cleaned
    # Content preserved
    assert "골프연습장 사용 협조 요청" in cleaned
