"""
ChromaDB 甕겸돧苑???쎈꽅????묐쓠.

????筌롫???怨쀬뵠?? c_id, domain, chunk_index, source_spec, token_count
鈺곌퀬?????醫롪텢???癒?땾(distance) ??釉?獄쏆꼹??
"""

from pathlib import Path
from typing import Optional
from observability.logger import logger

DEFAULT_CHROMA_DB_PATH = (Path(__file__).resolve().parents[3] / "chroma_db").resolve()


class ChromaStore:
    """
    ChromaDB ?뚎됱젂??CRUD ?????
    collection_name, persist_directory??config/settings.py?癒?퐣 雅뚯눘??
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        persist_directory: str = str(DEFAULT_CHROMA_DB_PATH),
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    def _get_collection(self):
        """?뚎됱젂??륁뱽 lazy ?λ뜃由?酉釉??獄쏆꼹??"""
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # cosine ?醫롪텢??????
            )
            logger.info(f"[ChromaStore] ?뚎됱젂??'{self.collection_name}' ?怨뚭퍙 ?袁⑥┷")
        except ImportError:
            raise ImportError("[ChromaStore] chromadb ???텕筌왖揶쎛 ?袁⑹뒄??몃빍??")
        return self._collection

    def save(
        self,
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
        documents: list[str],
    ) -> None:
        """筌?寃??袁⑥퓢??묐８李???怨쀬뵠?嫄붾９?앲눧紐꾩뱽 ChromaDB??????"""
        col = self._get_collection()
        col.upsert(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=documents,
        )
        logger.info(f"[ChromaStore] {len(ids)}揶?筌?寃?upsert ?袁⑥┷")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        ?묒눖???袁⑥퓢??뱀몵嚥?top_k ?醫롪텢 筌?寃?野꺜??

        獄쏆꼹???類ㅻ뻼 (chromadb ?癒?궚):
        {
            "ids": [[...]],
            "documents": [[...]],
            "metadatas": [[...]],
            "distances": [[...]]   ??cosine distance (?????롮쨯 ?醫롪텢)
        }
        """
        col = self._get_collection()
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def count(self) -> int:
        """?뚎됱젂??뤿퓠 ???貫留??袁⑷퍥 筌?寃???獄쏆꼹??"""
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """?뚎됱젂???袁⑷퍥 ????(???꾤빊???????."""
        if self._client:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.warning(f"[ChromaStore] Collection '{self.collection_name}' deleted")
