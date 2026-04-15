"""
JSON 원천 데이터 로더.

원천 데이터 형식:
{
    "c_id": str,
    "domain": int,
    "source": int,
    "source_spec": str,
    "creation_year": str,
    "content": str
}

domain 코드: 17 = 내과
"""

import json
import os

from domain.models.source_document import SourceDocument
from observability.logger import logger


class JSONLoader:
    """JSON 형태의 원천 데이터를 로드하고 domain 필터링을 수행하는 클래스."""

    DOMAIN_MAP = {
        17: "내과",
        # 향후 카테고리 확장 시 여기에 추가
        # 예: 1: "외과", 2: "피부과"
    }

    def __init__(self, target_domain: int = 17):
        self.target_domain = target_domain

    def load_file(self, path: str) -> list[SourceDocument]:
        """단일 JSON 파일을 로드하고 target_domain 필터링 후 SourceDocument 리스트 반환."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 단일 dict 또는 list[dict] 모두 대응
        if isinstance(data, dict):
            data = [data]

        filtered = [item for item in data if item.get("domain") == self.target_domain]
        docs = []
        for item in filtered:
            try:
                docs.append(SourceDocument(**item))
            except Exception as e:
                logger.warning(f"[JSONLoader] 파싱 실패 (c_id={item.get('c_id', '?')}): {e}")
        return docs

    def load_dir(self, dir_path: str, recursive: bool = True) -> list[SourceDocument]:
        """디렉터리 내 모든 JSON 파일을 로드하여 통합 반환."""
        results: list[SourceDocument] = []
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                if fname.endswith(".json"):
                    full_path = os.path.join(root, fname)
                    try:
                        loaded = self.load_file(full_path)
                        results.extend(loaded)
                        logger.info(f"[JSONLoader] {fname}: {len(loaded)}건 로드")
                    except Exception as e:
                        logger.error(f"[JSONLoader] 로드 실패: {full_path} — {e}")
            if not recursive:
                break
        logger.info(f"[JSONLoader] 총 {len(results)}건 로드 완료 (domain={self.target_domain})")
        return results
