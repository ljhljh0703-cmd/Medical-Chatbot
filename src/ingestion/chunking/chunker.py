"""
원천 데이터 청크닝 모듈.

전략:
1. 수자 접두사(\'1. \', \'2. \') 기준으로 의미 단위 우선 분할
2. 단위어 수 max_tokens 초과 시 미끄러지기(sliding window) 방식으로 재분할
3. 각 청크에 c_id, domain, chunk_index, source_spec 메타데이터 첨부
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """Chunker가 반환하는 단위 데이터 클래스."""
    text: str
    c_id: str
    domain: int
    chunk_index: int
    source_spec: Optional[str] = None
    token_count: int = 0


class Chunker:
    """원천 데이터 content를 의미 단위로 나누는 클래스."""

    # "1. ", "12. " 시작하는 줄 구분자
    _SECTION_SPLIT = re.compile(r"(?=\n\s*\d+\.\s)")

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        """
        max_tokens: 첩크 최대 토큰 수 (소박한 관련 혐정: 1토큰 ≈ 4문자)
        overlap_tokens: 재분할 시 이전 청크와 겨치는 토큰 수
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._chars_per_token = 4  # 한국어 기준 근사치

    def _approx_tokens(self, text: str) -> int:
        return max(1, len(text) // self._chars_per_token)

    def _split_long(self, text: str) -> list[str]:
        """싱글 직의 세그먼트가 max_tokens를 초과할 경우 토큰 단위로 재분할."""
        max_chars = self.max_tokens * self._chars_per_token
        overlap_chars = self.overlap_tokens * self._chars_per_token
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end].strip())
            start = end - overlap_chars
            if start >= len(text):
                break
        return [c for c in chunks if c]

    def chunk(self, content: str, c_id: str, domain: int, source_spec: Optional[str] = None) -> list[Chunk]:
        """
        content를 Chunk 리스트로 분할.
        c_id, domain, source_spec은 메타데이터로 저장.
        """
        if not content or not content.strip():
            return []

        # 1단계: 수자 접두사 기준 의미 단위 분할
        raw_sections = self._SECTION_SPLIT.split(content)
        raw_sections = [s.strip() for s in raw_sections if s.strip()]

        # 2단계: max_tokens 초과 시 재분할
        segments: list[str] = []
        for section in raw_sections:
            if self._approx_tokens(section) <= self.max_tokens:
                segments.append(section)
            else:
                segments.extend(self._split_long(section))

        # 3단계: Chunk 객체 조립
        return [
            Chunk(
                text=seg,
                c_id=c_id,
                domain=domain,
                chunk_index=i,
                source_spec=source_spec,
                token_count=self._approx_tokens(seg),
            )
            for i, seg in enumerate(segments)
        ]

    def chunk_doc(self, doc) -> list[Chunk]:
        """
        SourceDocument 또는 dict를 받아 chunk() 수행.
        doc: SourceDocument 인스턴스 또는 dict
        """
        if hasattr(doc, "content"):
            return self.chunk(
                content=doc.content,
                c_id=doc.c_id,
                domain=doc.domain,
                source_spec=doc.source_spec,
            )
        return self.chunk(
            content=doc.get("content", ""),
            c_id=doc.get("c_id", ""),
            domain=doc.get("domain", 0),
            source_spec=doc.get("source_spec"),
        )
