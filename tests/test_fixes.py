"""
Focused pytest tests for the AI Data Copilot fixes.
Covers: router wiring, report XAI format, NL-to-SQL, notebook import, DB connector safety.
"""
# ── 1. API route wiring ──────────────────────

class TestRouterWiring:
    """Verify all extra routers are actually registered on the FastAPI app."""

    def _paths(self):
        from backend.main import app
        return {route.path for route in app.routes}

    def test_database_router_included(self):
        paths = self._paths()
        assert "/api/database/schema" in paths
        assert "/api/database/query" in paths
        assert "/api/database/ask" in paths

    def test_memory_router_included(self):
        paths = self._paths()
        assert "/api/memory/save" in paths
        assert "/api/memory/load/{session_id}" in paths
        assert "/api/memory/list" in paths
        assert "/api/memory/latest" in paths

    def test_notebook_router_included(self):
        paths = self._paths()
        assert "/api/notebook/generate" in paths
        assert "/api/notebook/download" in paths

    def test_report_router_included(self):
        assert "/api/report/generate" in self._paths()

    def test_storage_router_included(self):
        paths = self._paths()
        assert "/api/storage/metrics" in paths
        assert "/api/storage/objects" in paths

    def test_presentation_router_included(self):
        assert "/api/presentation/generate" in self._paths()

    def test_no_double_api_prefix(self):
        paths = self._paths()
        assert not any(path.startswith("/api/api/") for path in paths)


# ── 2. Report generator XAI format ───────────

class TestReportGeneratorXAI:
    """report_generator.py must handle both dict and list formats for feature_importance."""

    def _call(self, xai_results):
        from backend.services.report_generator import generate_report
        return generate_report(
            dataset_summary={"rows": 100, "columns": 5, "features": ["a", "b"]},
            analysis_results={},
            visualization_summaries={},
            ml_results={"best_model": "RF", "metrics": {"accuracy": 0.9}},
            xai_results=xai_results,
            recommendations={},
        )

    def test_xai_dict_format(self):
        result = self._call({"feature_importance": {"age": 0.6, "income": 0.4}})
        assert "age" in result["markdown"]
        assert "0.6" in result["markdown"]

    def test_xai_list_format(self):
        result = self._call({
            "feature_importance": [
                {"feature": "age", "importance": 0.65},
                {"feature": "income", "importance": 0.35},
            ]
        })
        assert "age" in result["markdown"]
        assert "0.65" in result["markdown"]

    def test_xai_none(self):
        result = self._call(None)
        assert "Model explanations not provided" in result["markdown"]

    def test_xai_empty(self):
        result = self._call({})
        assert "Model explanations not provided" in result["markdown"]

    def test_xai_malformed_list_does_not_crash(self):
        result = self._call({"feature_importance": ["bad-row", {"importance": 0.4}]})
        assert "Feature importance format not recognized" in result["markdown"]


# ── 3. Full analysis runner with report ──────

class TestFullAnalysisRunner:
    """run_full_analysis with generate_report_flag=True must not crash."""

    def test_full_analysis_with_report(self):
        import pandas as pd
        from backend.services.full_analysis_runner import run_full_analysis

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = run_full_analysis(df, target_column="y", generate_report_flag=True)
        assert "report" in result
        assert "markdown" in result["report"]
        assert "## 7. Model Explanation" in result["report"]["markdown"]


# ── 4. NL-to-SQL grouped aggregate ──────────

class TestNLtoSQL:
    """Grouped aggregates must produce GROUP BY SQL."""

    SCHEMA = {
        "sales": [
            {"column": "region", "type": "TEXT"},
            {"column": "revenue", "type": "INTEGER"},
            {"column": "category", "type": "TEXT"},
        ]
    }

    def test_sum_by_region(self):
        from backend.services.nl_to_sql import nl_to_sql
        result = nl_to_sql("sum revenue by region", self.SCHEMA)
        sql = result["sql"].upper()
        assert "GROUP BY" in sql, f"Expected GROUP BY, got: {result['sql']}"
        assert "SUM" in sql

    def test_total_sales_by_category(self):
        from backend.services.nl_to_sql import nl_to_sql
        result = nl_to_sql("total revenue by category", self.SCHEMA)
        sql = result["sql"].upper()
        assert "GROUP BY" in sql, f"Expected GROUP BY, got: {result['sql']}"

    def test_revenue_by_region_implicit(self):
        from backend.services.nl_to_sql import nl_to_sql
        result = nl_to_sql("revenue by region", self.SCHEMA)
        sql = result["sql"].upper()
        assert "GROUP BY" in sql, f"Expected GROUP BY, got: {result['sql']}"

    def test_plain_sum_no_group(self):
        from backend.services.nl_to_sql import nl_to_sql
        result = nl_to_sql("sum of revenue", self.SCHEMA)
        sql = result["sql"].upper()
        assert "SUM" in sql
        assert "GROUP BY" not in sql, f"Plain sum should NOT have GROUP BY, got: {result['sql']}"


# ── 5. Notebook generator import ─────────────

class TestNotebookGenerator:
    """Verify nbformat is importable and the generator runs."""

    def test_nbformat_import(self):
        import nbformat  # noqa: F401

    def test_generate_notebook(self, tmp_path):
        from backend.services.notebook_generator import generate_analysis_notebook

        bundle = {
            "dataset_name": "test.csv",
            "dataset_summary": {"rows": 10, "columns": 3, "features": ["a"]},
        }
        out = tmp_path / "test.ipynb"
        path = generate_analysis_notebook(bundle, str(out))
        assert out.exists()
        assert path == str(out)


# ── 6. Database connector safety ─────────────

class TestDatabaseConnectorSafety:
    """Only safe SELECT queries should be allowed."""

    def test_select_is_safe(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("SELECT * FROM users") is True

    def test_drop_is_blocked(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("DROP TABLE users") is False

    def test_delete_is_blocked(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("DELETE FROM users WHERE 1=1") is False

    def test_update_is_blocked(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("UPDATE users SET name='x'") is False

    def test_insert_is_blocked(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("INSERT INTO users VALUES (1)") is False

    def test_select_with_hidden_drop_is_blocked(self):
        from backend.services.database_connector import DatabaseConnector
        assert DatabaseConnector.is_safe_query("SELECT 1; DROP TABLE users") is False

    def test_execute_safe_select_and_blocked_sql(self, tmp_path):
        import sqlite3
        from backend.services.database_connector import DatabaseConnector

        db_path = tmp_path / "demo.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE sales (region TEXT, revenue INTEGER)")
        conn.executemany("INSERT INTO sales VALUES (?, ?)", [("North", 100), ("South", 50)])
        conn.commit()
        conn.close()

        db = DatabaseConnector(db_type="sqlite", db_path=str(db_path))
        db.connect()
        try:
            result = db.execute_safe("SELECT region, revenue FROM sales")
            blocked = db.execute_safe("DELETE FROM sales")
        finally:
            db.close()

        assert result["row_count"] == 2
        assert "error" in blocked


# ── 7. Source validation and document comparison ─────────────

class TestSourceValidation:
    """Different files can reuse local chunk ids; citations must not collapse them."""

    def test_duplicate_chunk_ids_across_files_are_kept(self):
        from backend.rag.source_validator import build_source_citations

        chunks = [
            {
                "file_id": "file-a",
                "filename": "same.csv",
                "chunk_id": "0",
                "text": "A text",
                "rerank_score": 0.8,
                "rank": 1,
            },
            {
                "file_id": "file-b",
                "filename": "same.csv",
                "chunk_id": "0",
                "text": "B text",
                "rerank_score": 0.7,
                "rank": 2,
            },
        ]

        citations = build_source_citations(chunks, [])
        assert len(citations) == 2
        assert {c["file_id"] for c in citations} == {"file-a", "file-b"}


class TestDocumentComparatorFallback:
    """Comparison should remain useful when the LLM returns malformed JSON."""

    def test_llm_parse_error_returns_partial_comparison(self, monkeypatch):
        from backend.rag import document_comparator

        chunks = [
            {
                "file_id": "file-a",
                "filename": "same.csv",
                "chunk_id": "0",
                "text": "Shared Titanic dataset columns: PassengerId, Survived, Pclass.",
                "final_score": 0.9,
                "rerank_score": 0.8,
                "rank": 1,
            },
            {
                "file_id": "file-b",
                "filename": "same.csv",
                "chunk_id": "0",
                "text": "Shared Titanic dataset columns: PassengerId, Survived, Pclass.",
                "final_score": 0.9,
                "rerank_score": 0.75,
                "rank": 2,
            },
        ]

        monkeypatch.setattr(
            document_comparator,
            "retrieve_context",
            lambda question, file_ids=None: {"chunks": chunks, "status": "strong_context"},
        )
        monkeypatch.setattr(document_comparator, "call_llm", lambda *args, **kwargs: "not-json")

        result = document_comparator.compare_files("Compare these files", ["file-a", "file-b"])

        assert result["status"] == "partial_success"
        assert result["fallback_used"] is True
        assert result["common_points"]
        assert len(result["sources"]) == 2


# ── 8. PDF extraction ────────────────────────

class TestPDFExtraction:
    """PDF extraction should handle native text and support OCR fallback wiring."""

    def test_extract_native_pdf_text(self, tmp_path):
        import fitz
        from backend.rag.document_loader import extract_text

        pdf_path = tmp_path / "native.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "AI Data Copilot PDF extraction works.")
        doc.save(pdf_path)
        doc.close()

        text = extract_text(str(pdf_path), "native.pdf")
        assert "AI Data Copilot PDF extraction works" in text

    def test_ocr_fallback_is_used_when_native_text_is_empty(self, monkeypatch, tmp_path):
        from backend.rag import document_loader

        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

        monkeypatch.setattr(document_loader, "_extract_pdf_with_pypdf", lambda path: "")
        monkeypatch.setattr(document_loader, "_extract_pdf_with_pymupdf", lambda path: "")
        monkeypatch.setattr(document_loader, "_extract_pdf_with_ocr", lambda path: "OCR extracted text")

        assert document_loader.extract_pdf_text(str(pdf_path)) == "OCR extracted text"


# ── 9. RAG/indexing integration resilience ────────────────────────

class TestFileIntelligenceFallback:
    """File intelligence must not block indexing when Ollama/LLM is unavailable."""

    def test_generate_file_intelligence_survives_llm_outage(self, monkeypatch):
        from backend.rag import file_summarizer

        def fail_llm(*args, **kwargs):
            raise RuntimeError("ollama unavailable")

        monkeypatch.setattr(file_summarizer, "call_llm", fail_llm)

        result = file_summarizer.generate_file_intelligence(
            "AI Data Copilot extracts PDF text, chunks it, and indexes it for RAG."
        )

        assert result["summary"].startswith("Aperçu extrait automatiquement")
        assert result["tags"] == ["document"]
        assert result["suggested_questions"]


class TestRAGLLMFailureHandling:
    """RAG should return a degraded response with sources instead of raising."""

    def test_rag_agent_returns_sources_when_llm_fails(self, monkeypatch):
        from backend.agents import rag_agent

        monkeypatch.setattr(
            rag_agent,
            "call_llm",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ollama down")),
        )

        chunks = [{
            "file_id": "file-a",
            "filename": "scan.pdf",
            "chunk_id": "0",
            "text": "OCR extracted content about quarterly revenue and risks.",
            "rerank_score": 0.8,
            "rank": 1,
        }]

        result = rag_agent.run_rag_agent("What risks are mentioned?", chunks)

        assert result["llm_error"] is True
        assert result["grounded"] is False
        assert len(result["sources"]) == 1
        assert "LLM" in result["answer"]


class TestHybridSearchFallback:
    """Hybrid search should keep working when one retrieval backend fails."""

    def test_vector_failure_falls_back_to_keyword(self, monkeypatch):
        from backend.rag import hybrid_search

        monkeypatch.setattr(
            hybrid_search,
            "vector_search",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector unavailable")),
        )
        monkeypatch.setattr(
            hybrid_search.keyword_store,
            "search",
            lambda *args, **kwargs: [{
                "file_id": "file-a",
                "filename": "doc.txt",
                "chunk_id": "0",
                "text": "keyword result",
                "score": 2.0,
            }],
        )

        results = hybrid_search.run_hybrid_search("keyword", top_k=5)

        assert len(results) == 1
        assert results[0]["search_type"] == "keyword"
        assert results[0]["final_score"] == 0.3


class TestDatasetRAGResilience:
    """Dataset background indexing/retrieval should degrade when embeddings are down."""

    def test_index_session_data_returns_false_on_embedding_failure(self, monkeypatch):
        import pandas as pd
        from backend.core import indexer

        class BrokenIndexer:
            def __init__(self, session_id):
                raise RuntimeError("ollama embeddings down")

        monkeypatch.setattr(indexer, "DataIndexer", BrokenIndexer)

        result = indexer.index_session_data(
            "session-a",
            pd.DataFrame({"x": [1]}),
            {"columns": ["x"], "shape": (1, 1)},
        )

        assert result is False

    def test_retrieve_hybrid_context_returns_empty_on_retriever_failure(self, monkeypatch):
        from backend.core import retrieval

        class BrokenRetriever:
            def __init__(self, session_id):
                raise RuntimeError("chroma unavailable")

        monkeypatch.setattr(retrieval, "HybridRetriever", BrokenRetriever)

        assert retrieval.retrieve_hybrid_context("session-a", "question") == ""


class TestFileIndexEndpoint:
    """The /files/index endpoint should return structured API errors/results."""

    def test_index_file_success_without_real_vector_store(self, monkeypatch):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api import routes

        updates = {}

        monkeypatch.setattr(
            routes.file_registry,
            "get_all_files",
            lambda: {"file-a": {"filename": "scan.pdf", "indexed": False}},
        )
        monkeypatch.setattr(routes.file_manager, "get_file_path", lambda file_id, filename: "/tmp/scan.pdf")
        monkeypatch.setattr(routes.document_loader, "extract_text", lambda path, filename: "OCR text for indexing")
        monkeypatch.setattr(routes.chunker, "chunk_text", lambda text: ["OCR text for indexing"])
        monkeypatch.setattr(routes.vector_store, "add_chunks_to_store", lambda file_id, filename, chunks: None)
        monkeypatch.setattr(routes.keyword_index.keyword_store, "add_chunks", lambda chunks, metas: None)
        monkeypatch.setattr(
            routes,
            "generate_file_intelligence",
            lambda text, model_name=None: {
                "summary": "Indexed OCR text",
                "tags": ["document"],
                "key_topics": [],
                "suggested_questions": ["What is indexed?"],
            },
        )
        monkeypatch.setattr(
            routes.file_registry,
            "update_file_metadata",
            lambda file_id, metadata: updates.setdefault(file_id, metadata) or True,
        )

        response = TestClient(app).post("/api/files/index", params={"file_id": "file-a"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["indexed_chunks"] == 1
        assert payload["keyword_indexed"] is True
        assert updates["file-a"]["indexed"] is True

    def test_index_file_empty_extraction_returns_422(self, monkeypatch):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api import routes

        monkeypatch.setattr(
            routes.file_registry,
            "get_all_files",
            lambda: {"file-a": {"filename": "empty.pdf", "indexed": False}},
        )
        monkeypatch.setattr(routes.file_manager, "get_file_path", lambda file_id, filename: "/tmp/empty.pdf")
        monkeypatch.setattr(routes.document_loader, "extract_text", lambda path, filename: "")

        response = TestClient(app).post("/api/files/index", params={"file_id": "file-a"})

        assert response.status_code == 422
        assert "No extractable text" in response.json()["detail"]

    def test_delete_file_cleans_indexes(self, monkeypatch):
        from fastapi.testclient import TestClient
        from backend.main import app
        from backend.api import routes

        calls = []

        monkeypatch.setattr(
            routes.file_registry,
            "get_all_files",
            lambda: {"file-a": {"filename": "doc.txt", "indexed": True}},
        )
        monkeypatch.setattr(routes.vector_store, "delete_file_chunks", lambda file_id: calls.append(("vector", file_id)))
        monkeypatch.setattr(routes.keyword_index.keyword_store, "remove_file", lambda file_id: calls.append(("keyword", file_id)))
        monkeypatch.setattr(routes.file_manager, "delete_file", lambda file_id, filename: calls.append(("file", file_id)))
        monkeypatch.setattr(routes.file_registry, "delete_file_from_registry", lambda file_id: calls.append(("registry", file_id)))

        response = TestClient(app).delete("/api/files/file-a")

        assert response.status_code == 200
        assert response.json()["index_cleanup"] == {"vector": True, "keyword": True}
        assert calls == [
            ("vector", "file-a"),
            ("keyword", "file-a"),
            ("file", "file-a"),
            ("registry", "file-a"),
        ]


# ── 10. Storage Hub and voice presenter ────────────────────────

class TestStorageHub:
    """Storage Hub should work locally and keep PostgreSQL schema versioned."""

    def test_local_storage_hub_tracks_versions_and_metrics(self, tmp_path):
        from backend.services.storage_hub import LocalStorageHub

        hub = LocalStorageHub(tmp_path)
        hub.register_object(
            external_id="file-a",
            object_type="document",
            title="doc.txt",
            status="uploaded",
            tags=["txt"],
            metadata={"stage": "upload"},
        )
        hub.add_version(
            external_id="file-a",
            storage_uri="/tmp/doc.txt",
            content_hash="abc",
            size_bytes=12,
            metadata={"stage": "indexed"},
            created_by_agent="test",
        )
        hub.set_status("file-a", "indexed")

        item = hub.get_object("file-a")
        metrics = hub.metrics()

        assert item["status"] == "indexed"
        assert len(item["versions"]) == 1
        assert metrics["objects"] == 1
        assert metrics["by_type"]["document"] == 1

    def test_postgres_schema_file_exists(self):
        from pathlib import Path

        schema = Path("database/postgres_storage_schema.sql").read_text(encoding="utf-8")

        assert "CREATE TABLE IF NOT EXISTS adc_storage_objects" in schema
        assert "CREATE OR REPLACE FUNCTION adc_register_object" in schema
        assert "CREATE OR REPLACE FUNCTION adc_create_presentation" in schema


class TestVoicePresenter:
    """Voice presenter should generate and persist a speakable script."""

    def test_voice_presenter_fallback_persists_artifact(self, monkeypatch, tmp_path):
        from backend.services.storage_hub import LocalStorageHub
        from backend.services import voice_presenter

        hub = LocalStorageHub(tmp_path)

        monkeypatch.setattr(
            voice_presenter,
            "call_llm",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llm down")),
        )
        monkeypatch.setattr(voice_presenter, "get_storage_hub", lambda: hub)
        monkeypatch.setattr(
            voice_presenter.file_registry,
            "get_all_files",
            lambda: {
                "file-a": {
                    "filename": "report.pdf",
                    "indexed": True,
                    "summary": "Quarterly revenue increased.",
                    "tags": ["finance"],
                    "key_topics": ["Revenue"],
                }
            },
        )

        result = voice_presenter.generate_voice_presentation(
            topic="Quarterly report",
            file_ids=["file-a"],
            language="fr-FR",
        )

        assert result["speech_text"]
        assert result["segments"]
        assert result["metadata"]["fallback_used"] is True
        assert hub.get_object(result["storage_external_id"])["object_type"] == "voice_presentation"

    def test_voice_presenter_rejects_ungrounded_source_refs(self, monkeypatch, tmp_path):
        import json
        from backend.services.storage_hub import LocalStorageHub
        from backend.services import voice_presenter

        hub = LocalStorageHub(tmp_path)
        bad_llm_json = {
            "title": "Bad",
            "executive_summary": "Bad summary",
            "segments": [{
                "title": "Invented",
                "narration": "This adds facts not tied to the provided context.",
                "duration_hint_sec": 30,
                "source_refs": ["invented_source"],
            }],
            "closing": "End",
            "limitations": [],
        }

        monkeypatch.setattr(voice_presenter, "call_llm", lambda *args, **kwargs: json.dumps(bad_llm_json))
        monkeypatch.setattr(voice_presenter, "get_storage_hub", lambda: hub)
        monkeypatch.setattr(voice_presenter.file_registry, "get_all_files", lambda: {})

        result = voice_presenter.generate_voice_presentation(
            topic="Strict grounding",
            user_context="Only this context is allowed.",
        )

        assert result["metadata"]["fallback_used"] is True
        assert any("source_refs" in limitation for limitation in result["limitations"])

    def test_custom_voice_context_does_not_auto_include_documents(self, monkeypatch):
        from backend.services import voice_presenter

        monkeypatch.setattr(
            voice_presenter.file_registry,
            "get_all_files",
            lambda: {"file-a": {"filename": "doc.pdf", "indexed": True, "summary": "Should not appear"}},
        )

        context = voice_presenter.build_voice_context(
            file_ids=[],
            user_context="Only custom context.",
            include_all_files=False,
        )

        assert context["documents"] == []
        assert context["user_context"] == "Only custom context."

    def test_custom_voice_context_rejects_weak_grounding(self, monkeypatch, tmp_path):
        import json
        from backend.services.storage_hub import LocalStorageHub
        from backend.services import voice_presenter

        hub = LocalStorageHub(tmp_path)
        loose_llm_json = {
            "title": "Loose",
            "executive_summary": "Loose summary",
            "segments": [{
                "title": "Benefits",
                "narration": "This platform improves governance, search speed, team collaboration, enterprise scale, and regulatory readiness.",
                "duration_hint_sec": 30,
                "source_refs": ["user_context"],
            }],
            "closing": "End",
        }

        monkeypatch.setattr(voice_presenter, "call_llm", lambda *args, **kwargs: json.dumps(loose_llm_json))
        monkeypatch.setattr(voice_presenter, "get_storage_hub", lambda: hub)
        monkeypatch.setattr(voice_presenter.file_registry, "get_all_files", lambda: {})

        result = voice_presenter.generate_voice_presentation(
            topic="Strict custom",
            source_kind="custom",
            user_context="AI Data Copilot stores objects, versions, events, and voice presentations in PostgreSQL.",
        )

        assert result["metadata"]["fallback_used"] is True
        assert any("custom-context" in limitation for limitation in result["limitations"])
