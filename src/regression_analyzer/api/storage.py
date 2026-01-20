"""Simple file-based storage for uploads and results."""

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any


class Storage:
    """File-based storage manager for uploads and analysis results."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize storage.

        Args:
            base_dir: Base directory for storage. Defaults to /tmp/regression-analyzer
        """
        self.base_dir = Path(base_dir or "/tmp/regression-analyzer")
        self.uploads_dir = self.base_dir / "uploads"
        self.results_dir = self.base_dir / "results"

        # Create directories
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, content: bytes, filename: str) -> str:
        """Save uploaded file content.

        Args:
            content: File content as bytes
            filename: Original filename

        Returns:
            Upload ID (UUID)
        """
        upload_id = str(uuid.uuid4())
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / filename
        file_path.write_bytes(content)

        # Store metadata
        metadata = {
            "upload_id": upload_id,
            "filename": filename,
            "size": len(content),
        }
        (upload_dir / "metadata.json").write_text(json.dumps(metadata))

        return upload_id

    def get_upload_path(self, upload_id: str) -> Optional[Path]:
        """Get path to uploaded file.

        Args:
            upload_id: Upload ID

        Returns:
            Path to the uploaded file, or None if not found
        """
        upload_dir = self.uploads_dir / upload_id
        if not upload_dir.exists():
            return None

        # Find the data file (not metadata.json)
        for f in upload_dir.iterdir():
            if f.name != "metadata.json":
                return f

        return None

    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload metadata.

        Args:
            upload_id: Upload ID

        Returns:
            Metadata dict or None if not found
        """
        metadata_path = self.uploads_dir / upload_id / "metadata.json"
        if not metadata_path.exists():
            return None

        return json.loads(metadata_path.read_text())

    def get_job_dir(self, job_id: str) -> Path:
        """Get directory for a job's output.

        Args:
            job_id: Job ID

        Returns:
            Path to job directory (creates if needed)
        """
        job_dir = self.results_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_result(self, job_id: str, result: Dict[str, Any]) -> None:
        """Save analysis result.

        Args:
            job_id: Job ID
            result: Result dictionary
        """
        job_dir = self.get_job_dir(job_id)
        result_path = job_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, default=str))

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result.

        Args:
            job_id: Job ID

        Returns:
            Result dictionary or None if not found
        """
        result_path = self.results_dir / job_id / "result.json"
        if not result_path.exists():
            return None

        return json.loads(result_path.read_text())

    def get_chart_path(self, job_id: str, chart_name: str) -> Optional[Path]:
        """Get path to a chart image.

        Args:
            job_id: Job ID
            chart_name: Chart filename

        Returns:
            Path to chart or None if not found
        """
        # Charts are in the job's charts subdirectory
        chart_path = self.results_dir / job_id / "charts" / chart_name
        if chart_path.exists():
            return chart_path

        # Also check direct path (backwards compatibility)
        direct_path = self.results_dir / job_id / chart_name
        if direct_path.exists():
            return direct_path

        return None

    def list_charts(self, job_id: str) -> list[str]:
        """List available charts for a job.

        Args:
            job_id: Job ID

        Returns:
            List of chart filenames
        """
        charts_dir = self.results_dir / job_id / "charts"
        if not charts_dir.exists():
            return []

        return [f.name for f in charts_dir.iterdir() if f.suffix == ".png"]

    def cleanup_upload(self, upload_id: str) -> None:
        """Remove uploaded file and its directory.

        Args:
            upload_id: Upload ID
        """
        upload_dir = self.uploads_dir / upload_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

    def cleanup_job(self, job_id: str) -> None:
        """Remove job results and its directory.

        Args:
            job_id: Job ID
        """
        job_dir = self.results_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
