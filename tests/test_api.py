"""Tests for FastAPI REST endpoints."""

import csv
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from regression_analyzer.api import app
from regression_analyzer.api.storage import Storage


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_csv():
    """Create a sample CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "target"])
        for i in range(20):
            writer.writerow([i, i * 2, i * 3, i * 4 + 10])
        return Path(f.name)


@pytest.fixture
def storage():
    """Create a test storage instance."""
    return Storage(base_dir="/tmp/regression-analyzer-test")


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Health endpoint returns status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "llm_available" in data
        assert "providers" in data

    def test_providers_list(self, client):
        """Providers endpoint returns provider info."""
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
        # Should have at least openai, anthropic, google, groq
        provider_names = [p["name"] for p in data["providers"]]
        assert "openai" in provider_names
        assert "anthropic" in provider_names


class TestUploadEndpoint:
    """Tests for file upload endpoint."""

    def test_upload_csv(self, client, sample_csv):
        """Upload CSV file successfully."""
        with open(sample_csv, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert data["filename"] == "test.csv"
        assert data["status"] == "uploaded"
        assert data["size"] > 0

    def test_upload_invalid_extension(self, client):
        """Reject files with invalid extensions."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_upload_no_filename(self, client):
        """Reject uploads without filename."""
        response = client.post(
            "/api/v1/upload",
            files={"file": ("", b"data", "text/csv")},
        )
        # FastAPI should handle empty filename
        assert response.status_code in (400, 422)


class TestAnalyzeEndpoint:
    """Tests for analysis endpoint."""

    def test_analyze_uploaded_file(self, client, sample_csv):
        """Analyze an uploaded CSV file."""
        # Upload first
        with open(sample_csv, "rb") as f:
            upload_resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        upload_id = upload_resp.json()["upload_id"]

        # Analyze with LLM skipped for faster tests
        analyze_resp = client.post(
            "/api/v1/analyze",
            json={"upload_id": upload_id, "skip_llm": True},
        )
        assert analyze_resp.status_code == 200
        data = analyze_resp.json()
        assert "job_id" in data
        assert data["status"] == "completed"

    def test_analyze_with_target(self, client, sample_csv):
        """Analyze with specific target column."""
        # Upload first
        with open(sample_csv, "rb") as f:
            upload_resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        upload_id = upload_resp.json()["upload_id"]

        # Analyze with target specified
        analyze_resp = client.post(
            "/api/v1/analyze",
            json={"upload_id": upload_id, "target": "target", "skip_llm": True},
        )
        assert analyze_resp.status_code == 200
        data = analyze_resp.json()
        assert data["status"] == "completed"

    def test_analyze_missing_upload(self, client):
        """Return 404 for missing upload."""
        response = client.post(
            "/api/v1/analyze",
            json={"upload_id": "nonexistent-id", "skip_llm": True},
        )
        assert response.status_code == 404
        assert "Upload not found" in response.json()["detail"]


class TestResultsEndpoint:
    """Tests for results retrieval endpoint."""

    def test_get_results(self, client, sample_csv):
        """Retrieve analysis results."""
        # Upload and analyze
        with open(sample_csv, "rb") as f:
            upload_resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        upload_id = upload_resp.json()["upload_id"]

        analyze_resp = client.post(
            "/api/v1/analyze",
            json={"upload_id": upload_id, "skip_llm": True},
        )
        job_id = analyze_resp.json()["job_id"]

        # Get results
        results_resp = client.get(f"/api/v1/results/{job_id}")
        assert results_resp.status_code == 200
        data = results_resp.json()
        assert data["status"] == "completed"
        assert data["rows"] == 20
        assert data["columns"] == 4
        assert "statistics" in data

    def test_get_results_missing_job(self, client):
        """Return 404 for missing job."""
        response = client.get("/api/v1/results/nonexistent-job")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]


class TestChartsEndpoint:
    """Tests for charts endpoint."""

    def test_list_charts(self, client, sample_csv):
        """List available charts for a job."""
        # Upload and analyze
        with open(sample_csv, "rb") as f:
            upload_resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        upload_id = upload_resp.json()["upload_id"]

        analyze_resp = client.post(
            "/api/v1/analyze",
            json={"upload_id": upload_id, "skip_llm": True},
        )
        job_id = analyze_resp.json()["job_id"]

        # List charts
        charts_resp = client.get(f"/api/v1/charts/{job_id}")
        assert charts_resp.status_code == 200
        data = charts_resp.json()
        assert "charts" in data
        assert isinstance(data["charts"], list)

    def test_get_chart_missing(self, client, sample_csv):
        """Return 404 for missing chart."""
        # Upload and analyze first to get a valid job_id
        with open(sample_csv, "rb") as f:
            upload_resp = client.post(
                "/api/v1/upload",
                files={"file": ("test.csv", f, "text/csv")},
            )
        upload_id = upload_resp.json()["upload_id"]

        analyze_resp = client.post(
            "/api/v1/analyze",
            json={"upload_id": upload_id, "skip_llm": True},
        )
        job_id = analyze_resp.json()["job_id"]

        # Try to get non-existent chart
        response = client.get(f"/api/v1/charts/{job_id}/nonexistent.png")
        assert response.status_code == 404


class TestStorageModule:
    """Tests for storage module."""

    def test_save_and_get_upload(self, storage):
        """Save and retrieve uploaded file."""
        content = b"test,data\n1,2\n3,4"
        upload_id = storage.save_upload(content, "test.csv")

        # Verify path exists
        path = storage.get_upload_path(upload_id)
        assert path is not None
        assert path.exists()
        assert path.read_bytes() == content

        # Verify metadata
        metadata = storage.get_upload_metadata(upload_id)
        assert metadata["filename"] == "test.csv"
        assert metadata["size"] == len(content)

    def test_save_and_get_result(self, storage):
        """Save and retrieve analysis result."""
        result = {"status": "completed", "rows": 100, "columns": 5}
        job_id = "test-job-123"
        storage.save_result(job_id, result)

        # Retrieve
        retrieved = storage.get_result(job_id)
        assert retrieved == result

    def test_get_missing_result(self, storage):
        """Return None for missing result."""
        result = storage.get_result("nonexistent")
        assert result is None

    def test_cleanup_upload(self, storage):
        """Clean up uploaded file."""
        content = b"test data"
        upload_id = storage.save_upload(content, "test.csv")

        # Verify exists
        assert storage.get_upload_path(upload_id) is not None

        # Cleanup
        storage.cleanup_upload(upload_id)

        # Verify removed
        assert storage.get_upload_path(upload_id) is None
