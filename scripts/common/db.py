"""ChromaDB操作モジュール"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging


class VectorDB:
    """ベクトルデータベース操作クラス"""

    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.logger = logging.getLogger(__name__)

    def get_or_create_collection(self, collection_name: str, embedding_dim: int) -> chromadb.Collection:
        """コレクションを取得または作成"""
        try:
            collection = self.client.get_collection(name=collection_name)
            self.logger.info(f"既存のコレクション '{collection_name}' を使用します")
        except Exception:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": embedding_dim, "hnsw:space": "cosine"}
            )
            self.logger.info(f"新しいコレクション '{collection_name}' を作成しました")

        return collection

    def add_documents(self,
                     collection: chromadb.Collection,
                     embeddings: List[List[float]],
                     documents: List[str],
                     metadatas: List[Dict[str, Any]],
                     ids: List[str]) -> None:
        """ドキュメントをコレクションに追加"""
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        self.logger.info(f"{len(documents)}件のドキュメントを追加しました")

    def search(self,
               collection: chromadb.Collection,
               query_embedding: List[float],
               n_results: int = 5,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """類似度検索を実行"""
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=['embeddings', 'documents', 'metadatas', 'distances']
        )

        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1.0 - results['distances'][0][i]  # コサイン距離から類似度へ
            })

        return formatted_results

    def delete_collection(self, collection_name: str) -> None:
        """コレクションを削除"""
        try:
            self.client.delete_collection(name=collection_name)
            self.logger.info(f"コレクション '{collection_name}' を削除しました")
        except ValueError:
            self.logger.warning(f"コレクション '{collection_name}' は存在しません")

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """コレクションの統計情報を取得"""
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()

            sample = collection.get(limit=1)
            metadata_keys = set()
            if sample['metadatas']:
                metadata_keys = set(sample['metadatas'][0].keys())

            return {
                'collection_name': collection_name,
                'document_count': count,
                'metadata_fields': list(metadata_keys),
                'dimension': collection.metadata.get('dimension', 'unknown')
            }
        except ValueError:
            return {
                'collection_name': collection_name,
                'error': 'Collection not found'
            }

    def list_collections(self) -> List[str]:
        """すべてのコレクション名を取得"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

    def delete_by_metadata(self, collection: chromadb.Collection, where: Dict[str, Any]) -> int:
        """メタデータ条件に基づいてドキュメントを削除"""
        results = collection.get(
            where=where,
            include=[]  # IDsは常に返されるので指定不要
        )

        if results['ids']:
            collection.delete(ids=results['ids'])
            self.logger.info(f"{len(results['ids'])}件のドキュメントを削除しました")
            return len(results['ids'])
        else:
            self.logger.info("削除対象のドキュメントが見つかりませんでした")
            return 0

    def get_documents_by_metadata(self, collection: chromadb.Collection, where: Dict[str, Any]) -> List[Dict[str, Any]]:
        """メタデータ条件に基づいてドキュメントを取得"""
        results = collection.get(
            where=where,
            include=['ids', 'embeddings', 'documents', 'metadatas']
        )

        formatted_results = []
        for i in range(len(results['ids'])):
            formatted_results.append({
                'id': results['ids'][i],
                'document': results['documents'][i] if results['documents'] else None,
                'metadata': results['metadatas'][i] if results['metadatas'] else None
            })

        return formatted_results
