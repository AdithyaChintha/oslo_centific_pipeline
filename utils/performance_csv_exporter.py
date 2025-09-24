import csv
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class PipelineSingleCSVExporter:
    """Export all pipeline timing data to a single CSV file"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_timing_data(self, hierarchical_timing_data: Dict, session_id: str) -> str:
        """
        Export all timing data to a single CSV file

        Args:
            hierarchical_timing_data: Hierarchical timing data structure
            session_id: Session identifier for file naming

        Returns:
            Path to generated CSV file
        """
        filename = self.output_dir / f"{session_id}_timing_analysis.csv"

        # Extract session-level data for reuse
        session_data = self._extract_session_data(hierarchical_timing_data, session_id)

        # Generate all rows
        rows = []

        # Add session-level operations
        rows.extend(self._generate_session_rows(hierarchical_timing_data, session_data))

        # Add video-level operations
        rows.extend(self._generate_video_rows(hierarchical_timing_data, session_data))

        # Add model-level operations
        rows.extend(self._generate_model_rows(hierarchical_timing_data, session_data))

        # Write to CSV
        self._write_csv(filename, rows)

        return str(filename)

    def _extract_session_data(self, data: Dict, session_id: str) -> Dict:
        """Extract session-level data that will be repeated in all rows"""
        performance_metrics = data.get("performance_metrics", {})

        return {
            "session_id": session_id,
            "session_start_time": data.get("session_start_time", ""),
            "session_end_time": data.get("session_end_time", ""),
            "total_session_time": data.get("total_session_time", 0),
            "video_count": data.get("video_count", 0),
            "parallelization_efficiency": performance_metrics.get("parallelization_efficiency", 0),
            "slowest_model": performance_metrics.get("slowest_model", {}).get("name", ""),
            "fastest_model": performance_metrics.get("fastest_model", {}).get("name", "")
        }

    def _generate_session_rows(self, data: Dict, session_data: Dict) -> List[Dict]:
        """Generate rows for session-level operations"""
        rows = []

        # Session download operation
        download_timing = data.get("infrastructure_times", {}).get("session_download", {})
        if download_timing:
            rows.append({
                **session_data,
                "video_name": "",
                "operation_level": "session",
                "operation_type": "download",
                "shard_index": "",
                "view_name": "",
                "model_name": "session_download",
                "execution_time": download_timing.get("execution_time", 0),
                "start_time": download_timing.get("start_time", ""),
                "end_time": download_timing.get("end_time", ""),
                "timing_id": download_timing.get("timing_id", ""),
                "status": download_timing.get("status", ""),
                "shard_count": "",
                "total_video_time": "",
                "total_shard_time": "",
                "total_view_time": ""
            })

        return rows

    def _generate_video_rows(self, data: Dict, session_data: Dict) -> List[Dict]:
        """Generate rows for video-level operations"""
        rows = []

        for video in data.get("videos", []):
            video_name = video.get("video_name", "")
            total_video_time = video.get("total_video_time", 0)
            shard_count = video.get("shard_count", 0)

            # Process video-level operations
            video_processing = video.get("video_processing", {})

            for operation_type, operation_data in video_processing.items():
                if operation_type == "sharding":
                    # Handle nested sharding operations
                    for sharding_type, sharding_data in operation_data.items():
                        rows.append({
                            **session_data,
                            "video_name": video_name,
                            "operation_level": "video",
                            "operation_type": sharding_type,
                            "shard_index": "",
                            "view_name": "",
                            "model_name": "video_processing",
                            "execution_time": sharding_data.get("execution_time", 0),
                            "start_time": sharding_data.get("start_time", ""),
                            "end_time": sharding_data.get("end_time", ""),
                            "timing_id": sharding_data.get("timing_id", ""),
                            "status": sharding_data.get("status", ""),
                            "shard_count": shard_count,
                            "total_video_time": total_video_time,
                            "total_shard_time": "",
                            "total_view_time": ""
                        })
                else:
                    rows.append({
                        **session_data,
                        "video_name": video_name,
                        "operation_level": "video",
                        "operation_type": operation_type,
                        "shard_index": "",
                        "view_name": "",
                        "model_name": "video_processing",
                        "execution_time": operation_data.get("execution_time", 0),
                        "start_time": operation_data.get("start_time", ""),
                        "end_time": operation_data.get("end_time", ""),
                        "timing_id": operation_data.get("timing_id", ""),
                        "status": operation_data.get("status", ""),
                        "shard_count": shard_count,
                        "total_video_time": total_video_time,
                        "total_shard_time": "",
                        "total_view_time": ""
                    })

        return rows

    def _generate_model_rows(self, data: Dict, session_data: Dict) -> List[Dict]:
        """Generate rows for model-level operations"""
        rows = []

        for video in data.get("videos", []):
            video_name = video.get("video_name", "")
            total_video_time = video.get("total_video_time", 0)
            shard_count = video.get("shard_count", 0)

            for shard in video.get("shards", []):
                shard_index = shard.get("shard_index", 0)
                total_shard_time = shard.get("total_shard_time", 0)

                for view_name, view_data in shard.get("views", {}).items():
                    total_view_time = view_data.get("total_view_time", 0)

                    for model_name, model_data in view_data.get("models", {}).items():
                        rows.append({
                            **session_data,
                            "video_name": video_name,
                            "operation_level": "model",
                            "operation_type": "model_execution",
                            "shard_index": shard_index,
                            "view_name": view_name,
                            "model_name": model_name,
                            "execution_time": model_data.get("execution_time", 0),
                            "start_time": model_data.get("start_time", ""),
                            "end_time": model_data.get("end_time", ""),
                            "timing_id": model_data.get("timing_id", ""),
                            "status": model_data.get("status", ""),
                            "shard_count": shard_count,
                            "total_video_time": total_video_time,
                            "total_shard_time": total_shard_time,
                            "total_view_time": total_view_time
                        })

        return rows

    def _write_csv(self, filename: Path, rows: List[Dict]):
        """Write rows to CSV file"""
        if not rows:
            return

        fieldnames = [
            "session_id", "video_name", "operation_level", "operation_type",
            "shard_index", "view_name", "model_name", "execution_time",
            "start_time", "end_time", "timing_id", "status",
            "session_start_time", "session_end_time", "total_session_time",
            "video_count", "shard_count", "total_video_time",
            "total_shard_time", "total_view_time", "parallelization_efficiency",
            "slowest_model", "fastest_model"
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def export_timing_data_to_single_csv(hierarchical_timing_data: Dict, output_dir: str, session_id: str) -> str:
    """
    Export hierarchical timing data to a single CSV file

    Args:
        hierarchical_timing_data: Hierarchical timing data structure
        output_dir: Output directory for CSV file
        session_id: Session identifier

    Returns:
        Path to generated CSV file
    """
    # Add required imports
    import csv
    import os
    from pathlib import Path

    exporter = PipelineSingleCSVExporter(output_dir)
    return exporter.export_timing_data(hierarchical_timing_data, session_id)