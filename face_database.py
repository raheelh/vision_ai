import json
import os
import uuid
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings


class FaceDatabase:
    def __init__(
        self,
        persist_path: str = "db/chromadb",
        max_vectors_per_person: int = 5,
        match_threshold: float = 0.55,
    ):
        self.persist_path = persist_path
        self.max_vectors_per_person = max_vectors_per_person
        self.match_threshold = match_threshold
        os.makedirs(self.persist_path, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_path,
        )
        self.client = chromadb.Client(settings=settings)
        self.collection = self.client.get_or_create_collection(
            name="face_vectors",
            metadata={"distance_metric": "cosine"},
        )

        self.index_file = os.path.join(self.persist_path, "person_vectors.json")
        self.person_vectors: Dict[str, List[str]] = self._load_person_vectors()
        self.next_person_id = self._get_next_person_id()

    def _load_person_vectors(self) -> Dict[str, List[str]]:
        if not os.path.exists(self.index_file):
            return {}
        with open(self.index_file, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_person_vectors(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as fh:
            json.dump(self.person_vectors, fh, indent=2)

    def _get_next_person_id(self) -> int:
        if not self.person_vectors:
            return 1
        ids = [int(pid) for pid in self.person_vectors.keys() if pid.isdigit()]
        return max(ids, default=0) + 1

    def _prune_old_vectors(self, person_id: str) -> None:
        entries = self.person_vectors.get(person_id, [])
        while len(entries) > self.max_vectors_per_person:
            oldest_id = entries.pop(0)
            try:
                self.collection.delete(ids=[oldest_id])
            except Exception:
                pass
        self.person_vectors[person_id] = entries

    def identify(self, vector) -> Tuple[Optional[str], Optional[float]]:
        if len(self.person_vectors) == 0:
            return None, None
        try:
            results = self.collection.query(
                query_embeddings=[vector.tolist()],
                n_results=1,
                include=["distances", "metadatas"],
            )
        except Exception:
            return None, None

        if not results["distances"] or not results["distances"][0]:
            return None, None

        distance = float(results["distances"][0][0])
        metadata = results["metadatas"][0][0]
        person_id = metadata.get("person_id")
        if person_id and distance <= self.match_threshold:
            return str(person_id), distance
        return None, distance

    def register_vector(self, person_id: str, vector) -> None:
        vector_id = f"{person_id}_{uuid.uuid4().hex}"
        self.collection.add(
            ids=[vector_id],
            embeddings=[vector.tolist()],
            metadatas=[{"person_id": str(person_id)}],
            documents=[str(person_id)],
        )
        self.person_vectors.setdefault(str(person_id), []).append(vector_id)
        self._prune_old_vectors(str(person_id))
        self._save_person_vectors()

    def create_person(self, vector) -> str:
        person_id = str(self.next_person_id)
        self.next_person_id += 1
        self.register_vector(person_id, vector)
        return person_id

    def update_person(self, person_id: str, vector) -> None:
        self.register_vector(person_id, vector)
