# test_graceful_failure.py
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from unittest.mock import patch


class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {
            "prompt_eval_count": 10,
            "eval_count": 5,
            "eval_duration": 1_000_000_000,
        }


import rag

# Force every LLM call to return something that fails RAGAnswer validation
with patch.object(rag, "local_llm") as mock_llm:
    mock_llm.invoke.return_value = FakeResponse("")  # blank -> always invalid

    answer, sources, run_metrics = rag.rag(
        "What is machine learning?", request_id="test-graceful-fail"
    )

    print("ANSWER:", answer)
    print("GRACEFUL FAILURE:", run_metrics["llm_section"]["graceful_failure"])
    print("ATTEMPTS:", run_metrics["llm_section"]["validation_attempts"])
    print("LLM CALLED:", mock_llm.invoke.call_count, "times")
