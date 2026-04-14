"""
지식당 구축 파이프라인 오케스트레이터.

Loader → Cleaner → Chunker → Embedder → ChromaStore

실행 예시:
    python -m ingestion.indexing.build_knowledge_base \
        --data_dir data/raw \
        --collection medical_inner
"""

import argparse
from ingestion.loaders.json_loader import JSONLoader
from ingestion.preprocess.cleaner import TextCleaner
from ingestion.chunking.chunker import Chunker
from retrieval.dense.embedder import Embedder
from retrieval.dense.chroma_store import ChromaStore
from observability.logger import logger


class KnowledgeBaseBuilder:
    """Ingestion 전체 파이프라인을 실행하는 디렉터 클래스."""

    def __init__(
        self,
        data_dir: str,
        collection_name: str = "medical_inner",
        target_domain: int = 17,
        max_tokens: int = 512,
        batch_size: int = 64,
    ):
        self.loader = JSONLoader(target_domain=target_domain)
        self.cleaner = TextCleaner()
        self.chunker = Chunker(max_tokens=max_tokens)
        self.embedder = Embedder()
        self.store = ChromaStore(collection_name=collection_name)
        self.data_dir = data_dir
        self.batch_size = batch_size

    def build(self) -> int:
        """전체 파이프라인 실행. 적재된 전체 청크 수 반환."""
        # 1. 로드
        logger.info(f"[Builder] 데이터 로드 시작: {self.data_dir}")
        docs = self.loader.load_dir(self.data_dir)
        logger.info(f"[Builder] 로드 완료: {len(docs)}건")

        # 2. 정제 + 청크닝
        all_chunks = []
        for doc in docs:
            doc.content = self.cleaner.clean(doc.content)
            chunks = self.chunker.chunk_doc(doc)
            all_chunks.extend(chunks)
        logger.info(f"[Builder] 청크 생성 완료: {len(all_chunks)}개")

        # 3. 임베딩 + 저장 (배치 단위)
        total_saved = 0
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "c_id": c.c_id,
                    "domain": c.domain,
                    "chunk_index": c.chunk_index,
                    "source_spec": c.source_spec or "",
                    "token_count": c.token_count,
                }
                for c in batch
            ]
            ids = [f"{c.c_id}__{c.chunk_index}" for c in batch]
            embeddings = self.embedder.embed_batch(texts)
            self.store.save(embeddings=embeddings, metadatas=metadatas, ids=ids, documents=texts)
            total_saved += len(batch)
            logger.info(f"[Builder] {total_saved}/{len(all_chunks)} 청크 저장 완료")

        logger.info(f"[Builder] 지식당 구축 완료. 전체 {total_saved}개 청크 적재.")
        return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="지식당 구축 파이프라인")
    parser.add_argument("--data_dir", required=True, help="원천 JSON 데이터 디렉터리")
    parser.add_argument("--collection", default="medical_inner", help="ChromaDB 콜렉션 이름")
    parser.add_argument("--domain", type=int, default=17)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder(
        data_dir=args.data_dir,
        collection_name=args.collection,
        target_domain=args.domain,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    count = builder.build()
    print(f"\n✅ 지식당 구축 완료: {count}개 청크 적재")
