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


class JSONLoader:
    """JSON 형태의 원천 데이터를 로드하고 domain 필터링을 수행하는 클래스."""

    DOMAIN_MAP = {
        17: "내과",
        # 향후 카테고리 확장 시 여기에 추가
        # 예: 1: "외과", 2: "피부과"
    }

    def __init__(self, target_domain: int = 17):
        self.target_domain = target_domain

    def load_file(self, path: str) -> list[dict]:
        """단일 JSON 파일을 로드하고 target_domain 필터링 후 반환."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 단일 dict 또는 list[dict] 모두 대응
        if isinstance(data, dict):
            data = [data]

        return [item for item in data if item.get("domain") == self.target_domain]

    def load_dir(self, dir_path: str, recursive: bool = True) -> list[dict]:
        """디렉터리 내 모든 JSON 파일을 로드하여 통합 반환."""
        results = []
        for root, _, files in os.walk(dir_path):
            for fname in files:
                if fname.endswith(".json"):
                    full_path = os.path.join(root, fname)
                    try:
                        results.extend(self.load_file(full_path))
                    except Exception as e:
                        print(f"[JSONLoader] 로드 실패: {full_path} — {e}")
            if not recursive:
                break
        return results
