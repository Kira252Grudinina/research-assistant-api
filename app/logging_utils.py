# app/logging_utils.py

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

LOG_PATH = os.path.join(os.path.dirname(__file__), "queries.log")


def log_query(
    question: str,
    max_papers: Optional[int],
    paper_ids: List[str],
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "max_papers": max_papers,
        "paper_ids": paper_ids,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
