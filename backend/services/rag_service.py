import json
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO, Dict, List, Optional
from uuid import uuid4

from chat_history_db import (
    DEFAULT_PROJECT_DESCRIPTION,
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    HistoryStore,
    create_history_store,
)
from chunker import SmartChunker
from config_loader import Config
from embedding_factory import EmbeddingFactory
from generator import GeneratorFactory
from retriever import AdvancedRetriever
from tools import ToolRegistry
from vector_store import VectorStore, create_vector_store

logger = logging.getLogger(__name__)

_service_instance: Optional["RAGApplication"] = None
_service_lock = Lock()


class RAGApplication:
    def __init__(self, root_path: Path, history_store: Optional[HistoryStore] = None):
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
        self.vector_store_backend = str(self.config.get("vector_store.backend", "qdrant"))

        self.chunker = SmartChunker(
            chunk_size=int(self.config.get("system.chunk_size", 256)),
            overlap=int(self.config.get("system.overlap", 64)),
        )
        self.embedder = self._create_embedder()
        self.vector_store: VectorStore = create_vector_store(
            backend=self.vector_store_backend,
            dimension=self.embedder.dimension,
            persist_path=str(self._resolve_data_path(self.config.get("vector_store.persist_path", "./vector_db"))),
            qdrant_path=str(
                self._resolve_data_path(
                    self.config.get("vector_store.qdrant.path", "./vector_db/qdrant")
                )
            ),
            qdrant_collection_name=str(
                self.config.get("vector_store.qdrant.collection_name", "ragagument")
            ),
            qdrant_url=str(self.config.get("vector_store.qdrant.url", "")).strip() or None,
            qdrant_api_key_env=str(
                self.config.get("vector_store.qdrant.api_key_env", "QDRANT_API_KEY")
            ),
            qdrant_prefer_grpc=bool(self.config.get("vector_store.qdrant.prefer_grpc", False)),
            qdrant_timeout=int(self.config.get("vector_store.qdrant.timeout", 30)),
        )
        self.retriever = AdvancedRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            llm_model=str(self.config.get("llm.model", "deepseek-chat")),
            config_path=str(self.root_path / "config.yaml"),
        )
        self.generator = self._create_generator()
        self.tools = ToolRegistry()
        chat_db_path = Path(os.getenv("RAG_CHAT_DB_PATH", str(self.root_path / "chat_history.db")))
        chat_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_store = history_store or create_history_store(db_path=str(chat_db_path))
        self.default_project_id = DEFAULT_PROJECT_ID
        self._migrate_legacy_default_project_files()
        self._restore_vector_store_if_needed()

    def get_health(self) -> Dict[str, Any]:
        all_documents = self.list_all_documents()
        return {
            "status": "ok",
            "app": "ragagument-api",
            "api_version": "0.2.0",
            "embedding_provider": self._embedding_provider(),
            "vector_store_backend": self.vector_store_backend,
            "llm_model": str(self.config.get("llm.model", "deepseek-chat")),
            "project_count": len(self.list_projects()),
            "document_count": len(all_documents),
            "indexed_chunk_count": sum(document.get("indexed_chunk_count", 0) for document in all_documents),
            "session_count": len(self.history_store.get_all_sessions()),
        }

    def get_settings(self) -> Dict[str, Any]:
        return {
            "llm_model": str(self.config.get("llm.model", "deepseek-chat")),
            "llm_provider": str(self.config.get("llm.provider", "deepseek")),
            "embedding_provider": self._embedding_provider(),
            "embedding_model": self._embedding_model(),
            "vector_store_backend": self.vector_store_backend,
            "top_k": int(self.config.get("system.top_k", 5)),
            "chunk_size": int(self.config.get("system.chunk_size", 256)),
            "overlap": int(self.config.get("system.overlap", 64)),
            "enable_fusion": bool(self.config.get("advanced.enable_rewriting", True)),
            "enable_hybrid_search": bool(self.config.get("advanced.enable_hybrid_search", False)),
            "index_batch_size": self.index_batch_size,
            "index_max_workers": self.index_max_workers,
            "upload_max_file_size_mb": self.upload_max_file_size_mb,
        }

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        for project_id, name, description, created_at, session_count in self.history_store.list_projects():
            documents = self.list_documents(project_id)
            projects.append(
                {
                    "project_id": project_id,
                    "name": name,
                    "description": description,
                    "created_at": created_at,
                    "session_count": session_count,
                    "document_count": len(documents),
                    "indexed_chunk_count": sum(document["indexed_chunk_count"] for document in documents),
                    "pending_document_count": sum(
                        1 for document in documents if document.get("processing_status") != "indexed"
                    ),
                }
            )
        return projects

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        project_id, project_name, project_description, created_at = self.history_store.create_project(name, description)
        self._project_upload_dir(project_id).mkdir(parents=True, exist_ok=True)
        return {
            "project_id": project_id,
            "name": project_name,
            "description": project_description,
            "created_at": created_at,
            "session_count": 0,
            "document_count": 0,
            "indexed_chunk_count": 0,
            "pending_document_count": 0,
        }

    def update_project(self, project_id: str, name: str, description: str = "") -> Dict[str, Any]:
        updated_project_id, project_name, project_description, created_at = self.history_store.update_project(
            project_id,
            name,
            description,
        )
        documents = self.list_documents(updated_project_id)
        return {
            "project_id": updated_project_id,
            "name": project_name,
            "description": project_description,
            "created_at": created_at,
            "session_count": len(self.list_sessions(updated_project_id)),
            "document_count": len(documents),
            "indexed_chunk_count": sum(document["indexed_chunk_count"] for document in documents),
            "pending_document_count": sum(
                1 for document in documents if document.get("processing_status") != "indexed"
            ),
        }

    def delete_project(self, project_id: str) -> Dict[str, Any]:
        resolved_project_id = self._require_project(project_id)
        if resolved_project_id == self.default_project_id:
            raise ValueError("The default project cannot be deleted.")

        documents = self.list_documents(resolved_project_id)
        deleted_document_count = len(documents)
        project_dir = self._project_upload_dir(resolved_project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)

        deleted_session_count, deleted_message_count = self.history_store.delete_project(resolved_project_id)
        rebuild = self.rebuild_vector_store()

        return {
            "deleted_project_id": resolved_project_id,
            "deleted_session_count": deleted_session_count,
            "deleted_message_count": deleted_message_count,
            "deleted_document_count": deleted_document_count,
            "reindexed_document_count": rebuild["indexed_documents"],
            "reindexed_chunk_count": rebuild["indexed_chunks"],
        }

    def list_all_documents(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for project in self.list_projects():
            documents.extend(self.list_documents(project["project_id"]))
        return documents

    def list_documents(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        resolved_project_id = self._require_project(project_id)
        documents: List[Dict[str, Any]] = []
        project_dir = self._project_upload_dir(resolved_project_id)
        if not project_dir.exists():
            return documents

        for file_path in sorted(project_dir.iterdir()):
            if not file_path.is_file() or file_path.name.endswith(".meta.json"):
                continue
            documents.append(self._load_document_record(file_path, resolved_project_id))
        return documents

    def save_upload_stream(
        self,
        original_filename: str,
        stream: BinaryIO,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_project_id = self._require_project(project_id)
        project_dir = self._project_upload_dir(resolved_project_id)
        project_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self._sanitize_filename(original_filename)
        stored_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{safe_name}"
        target_path = project_dir / stored_name
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
            project_id=resolved_project_id,
        )
        self._write_document_record(target_path, record)
        return record

    def delete_document(self, project_id: str, document_id: str) -> Dict[str, Any]:
        resolved_project_id = self._require_project(project_id)
        file_path = self._project_upload_dir(resolved_project_id) / document_id
        if not file_path.exists():
            raise ValueError(f"Document not found: {document_id}")

        meta_path = self._document_meta_path(file_path)
        if file_path.exists():
            file_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        rebuild = self.rebuild_vector_store()
        return {
            "project_id": resolved_project_id,
            "deleted_document_id": document_id,
            "reindexed_document_count": rebuild["indexed_documents"],
            "reindexed_chunk_count": rebuild["indexed_chunks"],
        }

    def index_documents(
        self,
        project_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        resolved_project_id = self._require_project(project_id)
        selected_ids = set(document_ids or [])
        indexed_documents = 0
        skipped_documents = 0
        indexed_chunks = 0
        updated_records: List[Dict[str, Any]] = []
        index_updated = False

        for record in self.list_documents(resolved_project_id):
            if selected_ids and record["id"] not in selected_ids:
                continue

            if record["processed"] and record.get("processing_status") == "indexed":
                skipped_documents += 1
                updated_records.append(record)
                continue

            file_path = self._project_upload_dir(resolved_project_id) / record["stored_filename"]
            try:
                chunk_count = self._index_document_record(
                    file_path=file_path,
                    record=record,
                    project_id=resolved_project_id,
                )
            except Exception as exc:
                logger.error("Failed to index %s: %s", record["stored_filename"], exc)
                skipped_documents += 1
                updated_records.append(self._load_document_record(file_path, resolved_project_id))
                continue

            indexed_documents += 1
            indexed_chunks += chunk_count
            index_updated = True
            updated_records.append(self._load_document_record(file_path, resolved_project_id))

        if index_updated:
            self.retriever.invalidate_cache()

        if not selected_ids:
            updated_records = self.list_documents(resolved_project_id)

        return {
            "project_id": resolved_project_id,
            "indexed_documents": indexed_documents,
            "indexed_chunks": indexed_chunks,
            "skipped_documents": skipped_documents,
            "documents": updated_records,
        }

    def start_index_job(
        self,
        project_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        resolved_project_id = self._require_project(project_id)
        requested_document_ids = list(document_ids or [])
        if not requested_document_ids:
            requested_document_ids = [
                document["id"]
                for document in self.list_documents(resolved_project_id)
                if not document.get("processed")
            ]

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "project_id": resolved_project_id,
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
            self._set_document_job(resolved_project_id, document_id, job_id, "queued")

        self.index_executor.submit(self._run_index_job, job_id, resolved_project_id, requested_document_ids or None)
        return dict(job)

    def get_index_job(self, job_id: str) -> Dict[str, Any]:
        with self.index_jobs_lock:
            job = self.index_jobs.get(job_id)
            if job is None:
                raise KeyError(f"Index job not found: {job_id}")
            return dict(job)

    def validate_llm(self) -> Dict[str, Any]:
        return self.generator.validate_connection()

    def create_session(self, project_id: Optional[str] = None) -> str:
        resolved_project_id = self._require_project(project_id)
        return self.history_store.start_new_session(resolved_project_id)

    def list_sessions(self, project_id: Optional[str] = None) -> List[Dict[str, str]]:
        resolved_project_id = self._require_project(project_id)
        sessions = []
        for session_id, session_project_id, title, start_time in self.history_store.get_all_sessions(resolved_project_id):
            sessions.append(
                {
                    "session_id": session_id,
                    "project_id": session_project_id,
                    "title": title,
                    "start_time": start_time,
                }
            )
        return sessions

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        messages = []
        for role, content, timestamp in self.history_store.get_session_history(session_id):
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
        project_id: Optional[str] = None,
        top_k: Optional[int] = None,
        use_fusion: Optional[bool] = None,
        num_variants: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        active_project_id = self._resolve_project_for_query(project_id, session_id)
        active_session_id = session_id or self.create_session(active_project_id)
        self.history_store.save_message(active_session_id, "user", query)

        tool_call = self.tools.route_query(query)
        response_text = ""
        sources: List[Dict[str, Any]] = []
        tool_name: Optional[str] = None

        if tool_call and self.config.get("advanced.enable_tools", True):
            tool_name = str(tool_call["name"])
            response_text = self.tools.execute_tool(tool_name, str(tool_call["input"]))
        else:
            effective_filter = dict(metadata_filter or {})
            effective_filter["project_id"] = active_project_id

            retrieved_chunks = self.retriever.retrieve(
                query=query,
                top_k=int(top_k or self.config.get("system.top_k", 5)),
                use_fusion=bool(
                    self.config.get("advanced.enable_rewriting", True)
                    if use_fusion is None
                    else use_fusion
                ),
                num_variants=int(num_variants or self.config.get("advanced.fusion_queries", 3)),
                metadata_filter=effective_filter,
            )
            response_text = "".join(self.generator.generate_response(query, retrieved_chunks))
            sources = self._serialize_sources(retrieved_chunks)

        self.history_store.save_message(active_session_id, "assistant", response_text)
        self._maybe_update_session_title(active_session_id)

        return {
            "session_id": active_session_id,
            "project_id": active_project_id,
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

    def _create_generator(self):
        return GeneratorFactory.from_config(self.config)

    def _resolve_data_path(self, configured_path: str) -> Path:
        path = Path(configured_path)
        if path.is_absolute():
            return path
        return (self.root_path / path).resolve()

    def _project_upload_dir(self, project_id: str) -> Path:
        return self.upload_dir / project_id

    def _document_meta_path(self, file_path: Path) -> Path:
        return file_path.parent / f"{file_path.name}.meta.json"

    def _build_document_record(
        self,
        original_filename: str,
        stored_name: str,
        size_bytes: int,
        project_id: str,
    ) -> Dict[str, Any]:
        return {
            "id": stored_name,
            "project_id": project_id,
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
            "parser_names": [],
            "detected_content_types": [],
            "detected_structural_roles": [],
            "parsed_element_count": 0,
        }

    def _load_document_record(self, file_path: Path, project_id: Optional[str] = None) -> Dict[str, Any]:
        resolved_project_id = project_id or self._infer_project_id_from_path(file_path)
        meta_path = self._document_meta_path(file_path)
        record = {
            "id": file_path.name,
            "project_id": resolved_project_id,
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
            "parser_names": [],
            "detected_content_types": [],
            "detected_structural_roles": [],
            "parsed_element_count": 0,
        }

        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                record.update(json.load(handle))

        record["project_id"] = record.get("project_id") or resolved_project_id

        if record.get("processed") and record.get("processing_status") == "uploaded":
            record["processing_status"] = "indexed"

        return record

    def _write_document_record(self, file_path: Path, record: Dict[str, Any]) -> None:
        meta_path = self._document_meta_path(file_path)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)

    def _index_document_record(self, file_path: Path, record: Dict[str, Any], project_id: str) -> int:
        record["processing_status"] = "indexing"
        record["last_error"] = None
        self._write_document_record(file_path, record)

        indexed_chunk_count = 0
        batch_seen = False
        parser_names = set()
        content_types = set()
        structural_roles = set()
        element_indexes = set()

        try:
            for batch in self.chunker.iter_document_chunk_batches(
                str(file_path),
                batch_size=self.index_batch_size,
            ):
                batch_seen = True
                for chunk in batch:
                    chunk.metadata.update(
                        {
                            "project_id": project_id,
                            "document_id": record["id"],
                            "original_filename": record["original_filename"],
                            "upload_time": record["upload_time"],
                            "filename": record["original_filename"],
                        }
                    )
                    parser_name = chunk.metadata.get("parser_name")
                    content_type = chunk.metadata.get("content_type")
                    structural_role = chunk.metadata.get("structural_role")
                    element_index = chunk.metadata.get("element_index")

                    if parser_name:
                        parser_names.add(str(parser_name))
                    if content_type:
                        content_types.add(str(content_type))
                    if structural_role:
                        structural_roles.add(str(structural_role))
                    if element_index is not None:
                        element_indexes.add(str(element_index))

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
            record["parser_names"] = sorted(parser_names)
            record["detected_content_types"] = sorted(content_types)
            record["detected_structural_roles"] = sorted(structural_roles)
            record["parsed_element_count"] = len(element_indexes)
            self._write_document_record(file_path, record)
            return indexed_chunk_count
        except Exception as exc:
            record["processed"] = False
            record["processing_status"] = "failed"
            record["last_error"] = str(exc)
            record["active_index_job_id"] = None
            self._write_document_record(file_path, record)
            raise

    def _set_document_job(
        self,
        project_id: str,
        document_id: str,
        job_id: Optional[str],
        status: Optional[str] = None,
    ) -> None:
        file_path = self._project_upload_dir(project_id) / document_id
        if not file_path.exists():
            return

        record = self._load_document_record(file_path, project_id)
        record["active_index_job_id"] = job_id
        if status:
            record["processing_status"] = status
        self._write_document_record(file_path, record)

    def _run_index_job(self, job_id: str, project_id: str, document_ids: Optional[List[str]]) -> None:
        with self.index_jobs_lock:
            job = self.index_jobs[job_id]
            job["status"] = "running"
            job["started_at"] = datetime.utcnow().isoformat()

        for document_id in document_ids or []:
            self._set_document_job(project_id, document_id, job_id, "indexing")

        try:
            result = self.index_documents(project_id, document_ids)
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
                self._set_document_job(project_id, document_id, None)

    def _maybe_update_session_title(self, session_id: str) -> None:
        history = self.history_store.get_session_history(session_id)
        user_messages = [message for message in history if message[0] == "user"]
        if len(user_messages) == 1:
            title = self.history_store.get_session_title_suggestion(session_id)
            self.history_store.update_session_title(session_id, title)

    def _serialize_sources(self, retrieved_chunks: List[Any]) -> List[Dict[str, Any]]:
        sources = []
        for result in retrieved_chunks:
            metadata = {
                key: self._serialize_metadata_value(value)
                for key, value in result.chunk.metadata.items()
            }
            sources.append(
                {
                    "filename": str(result.chunk.metadata.get("filename", "unknown")),
                    "score": float(result.score),
                    "snippet": result.chunk.content[:280],
                    "metadata": metadata,
                }
            )
        return sources

    def _serialize_metadata_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, (list, tuple, set)):
            return [self._serialize_metadata_value(item) for item in value]

        if isinstance(value, dict):
            return {
                str(key): self._serialize_metadata_value(item)
                for key, item in value.items()
            }

        return str(value)

    def _sanitize_filename(self, filename: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-")
        return sanitized or "upload.bin"

    def rebuild_vector_store(self) -> Dict[str, int]:
        logger.info("Rebuilding vector store from remaining indexed documents.")
        self.vector_store.reset(persist=True)

        indexed_documents = 0
        indexed_chunks = 0
        for project in self.list_projects():
            project_id = project["project_id"]
            for record in self.list_documents(project_id):
                if not record.get("processed") or record.get("processing_status") != "indexed":
                    continue

                file_path = self._project_upload_dir(project_id) / record["stored_filename"]
                if not file_path.exists():
                    continue

                try:
                    indexed_chunks += self._index_document_record(file_path, record, project_id)
                    indexed_documents += 1
                except Exception as exc:
                    logger.error("Failed to rebuild index for %s: %s", record["stored_filename"], exc)

        self.retriever.invalidate_cache()
        return {
            "indexed_documents": indexed_documents,
            "indexed_chunks": indexed_chunks,
        }

    def _embedding_provider(self) -> str:
        return str(self.config.get("embedding.provider", "ollama"))

    def _embedding_model(self) -> str:
        if self.config.get("embedding.provider"):
            provider = self._embedding_provider()
            if provider == "huggingface":
                return str(self.config.get("embedding.huggingface_model", "all-MiniLM-L6-v2"))
            return str(self.config.get("embedding.ollama_model", "mxbai-embed-large:latest"))
        return str(self.config.get("embedding.model", "mxbai-embed-large:latest"))

    def _require_project(self, project_id: Optional[str]) -> str:
        return self.history_store.ensure_project(project_id or self.default_project_id)

    def _resolve_project_for_query(self, project_id: Optional[str], session_id: Optional[str]) -> str:
        if session_id:
            session_project_id = self.history_store.get_session_project_id(session_id)
            if session_project_id is None:
                raise ValueError(f"Session not found: {session_id}")
            if project_id and project_id != session_project_id:
                raise ValueError("Session does not belong to the selected project.")
            return session_project_id

        return self._require_project(project_id)

    def _infer_project_id_from_path(self, file_path: Path) -> str:
        if file_path.parent == self.upload_dir:
            return self.default_project_id
        return file_path.parent.name

    def _migrate_legacy_default_project_files(self) -> None:
        """Move old flat uploaded_docs files into the default project folder."""
        default_dir = self._project_upload_dir(self.default_project_id)
        default_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.upload_dir.iterdir():
            if file_path == default_dir or file_path.is_dir():
                continue

            target_path = default_dir / file_path.name
            if target_path.exists():
                continue

            file_path.rename(target_path)

            if target_path.is_file() and not target_path.name.endswith(".meta.json"):
                record = self._load_document_record(target_path, self.default_project_id)
                if not record.get("project_id"):
                    record["project_id"] = self.default_project_id
                    self._write_document_record(target_path, record)

    def _restore_vector_store_if_needed(self) -> None:
        if self.vector_store.get_all_chunks():
            return

        indexed_records = []
        for project in self.list_projects():
            for record in self.list_documents(project["project_id"]):
                if record.get("processed") and record.get("processing_status") == "indexed":
                    indexed_records.append(record)

        if not indexed_records:
            return

        logger.info(
            "Vector store backend %s is empty. Rebuilding from %s indexed documents.",
            self.vector_store_backend,
            len(indexed_records),
        )
        self.rebuild_vector_store()


def get_rag_application() -> RAGApplication:
    global _service_instance
    if _service_instance is None:
        with _service_lock:
            if _service_instance is None:
                _service_instance = RAGApplication(root_path=Path(__file__).resolve().parents[2])
    return _service_instance
