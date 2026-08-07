"""
=============================================================================
Thiranex Solutions — Automated Scheduling & Monitoring Engine
Author: Google Deepmind Agentic AI Team
=============================================================================
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

def create_scheduled_job(job_name: str, frequency: str, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates automated scheduled job definition for Daily/Weekly/Monthly pipeline execution.
    """
    return {
        "job_id": f"JOB-{int(time.time())}",
        "job_name": job_name,
        "frequency": frequency,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Active / Scheduled",
        "next_run": f"Scheduled ({frequency})"
    }

def monitor_directory_for_new_files(watch_dir: str) -> List[str]:
    """
    Scans directory for uncleaned data files.
    """
    if not os.path.exists(watch_dir):
        return []
    
    supported_exts = (".csv", ".xlsx", ".xls", ".json", ".xml", ".parquet")
    new_files = []
    for root, dirs, files in os.walk(watch_dir):
        for f in files:
            if f.lower().endswith(supported_exts):
                new_files.append(os.path.join(root, f))
    return new_files
