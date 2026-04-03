import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Dict, List, Optional
from uuid import uuid4

from chat_history_db import ChatHistoryDB
from chunker import SmartChunker
from config_loader import Config
from embedding_factory import EmbeddingFactory
from generator import DeepSeekGenerator
from retriever import AdvancedRetriever
from tools import ToolRegistry
from vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

_service_instance: Optional["RAGApplication"] = None
_service_lock = Lock()


class RAGApplication:
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.config = Config(config_path=str(self.root_path / "config.yaml"))
        self.upload_dir = self.root_path / "uploaded_docs"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.upload_max_file_size_mb = int(self.config.get("uploads.max_file_size_mb", 250))
        self.index_batch_size = int(self.config.get("indexing.batch_size", 16))
        self.index_max_workers = int(self.config.get("indexing.max_workers", 2))
        self.index_executor = ThreadPoolExecutor(max_workers=self.index_max_workers, thread_name_prefix="rag-index")
        self.index_jobs: Dict[str, Dict[str, Any]] = {}
        self.index_jobs_lock = Lock()

        self.chunker = SmartChunker(
            chunk_size=int(self.config.get("system.chunk_size", 256)),
            overlap=int(self.config.get("system.overlap", 64)),
        )
        self.embedder = self._create_embedder()
        self.vector_store = FAISSVectorStore(
            dimension=self.embedder.dimension,
            persist_path=str(self._resolve_data_path(self.config.get("vector_store.persist_path", "./vector_db"))),
        )
        self.retriever = AdvancedRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm_model=str(self.config.get("llm.model", "deepseek-chat")),
            config_path=str(self.root_path / "config.yaml"),
        )
        self.generator = DeepSeekGenerator(
            model_name=str(self.config.get("llm.model", "deepseek-chat")),
            temperature=float(self.config.get("llm.temperature", 0.2)),
        )
        self.tools = ToolRegistry()
        chat_db_path = Path(os.getenv("RAG_CHAT_DB_PATH", str(self.root_path / "chat_history.db")))
        chat_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chat_db = ChatHistoryDB(db_path=str(chat_db_path))

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "app": "ragagument-api",
            "api_version": "0.1.0",
            "embedding_provider": self._embedding_provider(),
            "llm_model": str(self.config.get("llm.model", "deepseek-chat")),
            "document_count": len(self.list_documents()),
            "indexed_chunk_count": len(self.vector_store.get_all_chunks()),
            "session_count": len(self.chat_db.get_all_sessions()),
        }

    def get_settings(self) -> Dict[str, Any]:
        return {
            "llm_model": str(self.config.get("llm.model", "deepseek-chat")),
            "llm_provider": str(self.config.get("llm.provider", "deepseek")),
            "embedding_provider": self._embedding_provider(),
            "embedding_model": self._embedding_model(),
            "top_k": int(self.config.get("system.top_k", 5)),
            "chunk_size": int(self.config.get("system.chunk_size", 256)),
            "overlap": int(self.config.get("system.overlap", 64)),
            "enable_fusion": bool(self.config.get("advanced.enable_rewriting", True)),
            "enable_hybrid_search": bool(self.config.get("advanced.enable_hybrid_search", False)),
            "index_batch_size": self.index_batch_size,
            "index_max_workers": self.index_max_workers,
            "upload_max_file_size_mb": self.upload_max_file_size_mb,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for file_path in sorted(self.upload_dir.iterdir()):
            if not file_path.is_file() or file_path.name.endswith(".meta.json"):
                continue
            documents.append(self._load_document_record(file_path))
        return documents

    def save_upload_stream(self, original_filename: str, stream: BinaryIO) -> Dict[str, Any]:
        safe_name = self._sanitize_filename(original_filename)
        stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{safe_name}"
        target_path = self.upload_dir / stored_name
        max_bytes = self.upload_max_file_size_mb * 1024 * 1024
        size_bytes = 0

        stream.seek(0)
        try:
            with target_path.open("wb") as output_handle:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break

                    size_bytes += len(chunk)
                    if size_bytes > max_bytes:
                        raise ValueError(
                            f"File exceeds the configured upload limit of {self.upload_max_file_size_mb} MB."
                        )

                    output_handle.write(chunk)
        except Exception:
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            raise

        record = self._build_document_record(
            original_filename=original_filename,
            stored_name=stored_name,
            size_bytes=size_bytes,
        )
        self._write_document_record(target_path, record)
        return record

    def index_documents(self, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        selected_ids = set(document_ids or [])
        indexed_documents = 0
        skipped_documents = 0
        indexed_chunks = 0
        updated_records: List[Dict[str, Any]] = []
        index_updated = False

        for record in self.list_documents():
            if selected_ids and record["id"] not in selected_ids:
                continue

            if record["processed"] and record.get("processing_status") == "indexed":
                skipped_documents += 1
                updated_records.append(record)
                continue

            file_path = self.upload_dir / record["stored_filename"]
            try:
                chunk_count = self._index_document_record(file_path=file_path, record=record)
            except Exception as exc:
                logger.error("Failed to index %s: %s", record["stored_filename"], exc)
                skipped_documents += 1
                updated_records.append(self._load_document_record(file_path))
                continue

            indexed_documents += 1
            indexed_chunks += chunk_count
            index_updated = True
            updated_records.append(self._load_document_record(file_path))

        if index_updated:
            self.retriever.invalidate_cache()

        if not selected_ids:
            updated_records = self.list_documents()

        return {
            "indexed_documents": indexed_documents,
            "indexed_chunks": indexed_chunks,
            "skipped_documents": skipped_documents,
            "documents": updated_records,
        }

    def start_index_job(self, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        requested_document_ids = list(document_ids or [])
        if not requested_document_ids:
            requested_document_ids = [
                document["id"]
                for document in self.list_documents()
                if not document.get("processed")
            ]

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "document_ids": requested_document_ids,
            "submitted_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "skipped_documents": 0,
            "error": None,
        }

        with self.index_jobs_lock:
            self.index_jobs[job_id] = job

        for document_id in requested_document_ids:
            self._set_document_job(document_id, job_id, "queued")

        self.index_executor.submit(self._run_index_job, job_id, requested_document_ids or None)
        return dict(job)

    def get_index_job(self, job_id: str) -> Dict[str, Any]:
        with self.index_jobs_lock:
            job = self.index_jobs.get(job_id)
            if job is None:
                raise KeyError(f"Index job not found: {job_id}")
            return dict(job)

    def validate_llm(self) -> Dict[str, Any]:
        api_key = self.generator.client.api_key
        if not api_key:
            return {
                "provider": "deepseek",
                "configured": False,
                "valid": False,
                "message": "DEEPSEEK_API_KEY is not configured.",
            }

        try:
            self.generator.client.models.list()
            return {
                "provider": "deepseek",
                "configured": True,
                "valid": True,
                "message": "DeepSeek credentials validated successfully.",
            }
        except Exception as exc:
            message = str(exc)
            if "Authentication" in message or "invalid" in message.lower():
                message = "DeepSeek authentication failed. Check DEEPSEEK_API_KEY."
            return {
                "provider": "deepseek",
                "configured": True,
                "valid": False,
                "message": message,
            }

    def create_session(self) -> str:
        return self.chat_db.start_new_session()

    def list_sessions(self) -> List[Dict[str, str]]:
        sessions = []
        for session_id, title, start_time in self.chat_db.get_all_sessions():
            sessions.append(
                {
                    "session_id": session_id,
                    "title": title,
                    "start_time": start_time,
                }
            )
        return sessions

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        messages = []
        for role, content, timestamp in self.chat_db.get_session_history(session_id):
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                }
            )
        return messages

    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        use_fusion: Optional[bool] = None,
        num_variants: Optional[int] = None,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        active_session_id = session_id or self.create_session()
        self.chat_db.save_message(active_session_id, "user", query)

        tool_call = self.tools.route_query(query)
        response_text = ""
        sources: List[Dict[str, Any]] = []
        tool_name: Optional[str] = None

        if tool_call and self.config.get("advanced.enable_tools", True):
            tool_name = str(tool_call["name"])
            response_text = self.tools.execute_tool(tool_name, str(tool_call["input"]))
        else:
            retrieved_chunks = self.retriever.retrieve(
                query=query,
                top_k=int(top_k or self.config.get("system.top_k", 5)),
                use_fusion=bool(
                    self.config.get("advanced.enable_rewriting", True)
                    if use_fusion is None
                    else use_fusion
                ),
                num_variants=int(num_variants or self.config.get("advanced.fusion_queries", 3)),
                metadata_filter=metadata_filter,
            )
            response_text = "".join(self.generator.generate_response(query, retrieved_chunks))
            sources = self._serialize_sources(retrieved_chunks)

        self.chat_db.save_message(active_session_id, "assistant", response_text)
        self._maybe_update_session_title(active_session_id)

        return {
            "session_id": active_session_id,
            "response": response_text,
            "tool_name": tool_name,
            "sources": sources,
        }

    def _create_embedder(self) -> EmbeddingFactory:
        provider = self.config.get("embedding.provider")
        if provider:
            return EmbeddingFactory(
                provider=str(provider),
                huggingface_model=str(self.config.get("embedding.huggingface_model", "all-MiniLM-L6-v2")),
                ollama_model=str(self.config.get("embedding.ollama_model", "mxbai-embed-large:latest")),
            )
        return EmbeddingFactory(
            provider="ollama",
            ollama_model=str(self.config.get("embedding.model", "mxbai-embed-large:latest")),
        )

    def _resolve_data_path(self, configured_path: str) -> Path:
        path = Path(configured_path)
        if path.is_absolute():
            return path
        return (self.root_path / path).resolve()

    def _document_meta_path(self, file_path: Path) -> Path:
        return file_path.parent / f"{file_path.name}.meta.json"

    def _build_document_record(self, original_filename: str, stored_name: str, size_bytes: int) -> Dict[str, Any]:
        return {
            "id": stored_name,
            "original_filename": original_filename,
            "stored_filename": stored_name,
            "upload_time": datetime.utcnow().isoformat(),
            "size_bytes": size_bytes,
            "processed": False,
            "indexed_chunk_count": 0,
            "processing_status": "uploaded",
            "last_error": None,
            "last_indexed_at": None,
            "active_index_job_id": None,
        }

    def _load_document_record(self, file_path: Path) -> Dict[str, Any]:
        meta_path = self._document_meta_path(file_path)
        record = {
            "id": file_path.name,
            "original_filename": file_path.name,
            "stored_filename": file_path.name,
            "upload_time": datetime.utcfromtimestamp(file_path.stat().st_mtime).isoformat(),
            "size_bytes": int(file_path.stat().st_size),
            "processed": False,
            "indexed_chunk_count": 0,
            "processing_status": "uploaded",
            "last_error": None,
            "last_indexed_at": None,
            "active_index_job_id": None,
        }

        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                record.update(json.load(handle))

        if record.get("processed") and record.get("processing_status") == "uploaded":
            record["processing_status"] = "indexed"

        return record

    def _write_document_record(self, file_path: Path, record: Dict[str, Any]) -> None:
        meta_path = self._document_meta_path(file_path)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

    def _index_document_record(self, file_path: Path, record: Dict[str, Any]) -> int:
        record["processing_status"] = "indexing"
        record["last_error"] = None
        self._write_document_record(file_path, record)

        indexed_chunk_count = 0
        batch_seen = False

        try:
            for batch in self.chunker.iter_document_chunk_batches(
                str(file_path),
                batch_size=self.index_batch_size,
            ):
                batch_seen = True
                for chunk in batch:
                    chunk.metadata.update(
                        {
                            "document_id": record["id"],
                            "original_filename": record["original_filename"],
                            "upload_time": record["upload_time"],
                        }
                    )

                embedded_chunks = self.embedder.embed_chunks(batch)
                self.vector_store.add_chunks(embedded_chunks, persist=False)
                indexed_chunk_count += len(embedded_chunks)

            if not batch_seen:
                raise ValueError("No chunks were produced from the document.")

            self.vector_store.persist()
            record["processed"] = True
            record["indexed_chunk_count"] = indexed_chunk_count
            record["processing_status"] = "indexed"
            record["last_indexed_at"] = datetime.utcnow().isoformat()
            record["active_index_job_id"] = None
            self._write_document_record(file_path, record)
            return indexed_chunk_count
        except Exception as exc:
            record["processed"] = False
            record["processing_status"] = "failed"
            record["last_error"] = str(exc)
            record["active_index_job_id"] = None
            self._write_document_record(file_path, record)
            raise

    def _set_document_job(self, document_id: str, job_id: Optional[str], status: Optional[str] = None) -> None:
        file_path = self.upload_dir / document_id
        if not file_path.exists():
            return

        record = self._load_document_record(file_path)
        record["active_index_job_id"] = job_id
        if status:
            record["processing_status"] = status
        self._write_document_record(file_path, record)

    def _run_index_job(self, job_id: str, document_ids: Optional[List[str]]) -> None:
        with self.index_jobs_lock:
            job = self.index_jobs[job_id]
            job["status"] = "running"
            job["started_at"] = datetime.utcnow().isoformat()

        for document_id in document_ids or []:
            self._set_document_job(document_id, job_id, "indexing")

        try:
            result = self.index_documents(document_ids)
            with self.index_jobs_lock:
                job = self.index_jobs[job_id]
                job["status"] = "completed"
                job["finished_at"] = datetime.utcnow().isoformat()
                job["indexed_documents"] = result["indexed_documents"]
                job["indexed_chunks"] = result["indexed_chunks"]
                job["skipped_documents"] = result["skipped_documents"]
                job["error"] = None
        except Exception as exc:
            with self.index_jobs_lock:
                job = self.index_jobs[job_id]
                job["status"] = "failed"
                job["finished_at"] = datetime.utcnow().isoformat()
                job["error"] = str(exc)
        finally:
            for document_id in document_ids or []:
                self._set_document_job(document_id, None)

    def _maybe_update_session_title(self, session_id: str) -> None:
        history = self.chat_db.get_session_history(session_id)
        user_messages = [message for message in history if message[0] == "user"]
        if len(user_messages) == 1:
            title = self.chat_db.get_session_title_suggestion(session_id)
            self.chat_db.update_session_title(session_id, title)

    def _serialize_sources(self, retrieved_chunks: List[Any]) -> List[Dict[str, Any]]:
        sources = []
        for result in retrieved_chunks:
            metadata = {key: str(value) for key, value in result.chunk.metadata.items()}
            sources.append(
                {
                    "filename": str(result.chunk.metadata.get("filename", "unknown")),
                    "score": float(result.score),
                    "snippet": result.chunk.content[:280],
                    "metadata": metadata,
                }
            )
        return sources

    def _sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-")
        return sanitized or "upload.bin"

    def _embedding_provider(self) -> str:
        return str(self.config.get("embedding.provider", "ollama"))

    def _embedding_model(self) -> str:
        if self.config.get("embedding.provider"):
            provider = self._embedding_provider()
            if provider == "huggingface":
                return str(self.config.get("embedding.huggingface_model", "all-MiniLM-L6-v2"))
            return str(self.config.get("embedding.ollama_model", "mxbai-embed-large:latest"))
        return str(self.config.get("embedding.model", "mxbai-embed-large:latest"))


def get_rag_application() -> RAGApplication:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = RAGApplication(root_path=Path(__file__).resolve().parents[2])
    return _service_instance
