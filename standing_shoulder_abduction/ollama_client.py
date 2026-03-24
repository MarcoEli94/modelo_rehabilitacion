from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.ollama_client import ask_ollama as shared_ask_ollama
from api.ollama_client import prepare_diagnosis as shared_prepare_diagnosis

EXERCISE_ID = 9


def prepare_diagnosis(raw_prediction: Dict[str, Any]) -> Dict[str, Any]:
    return shared_prepare_diagnosis(raw_prediction, EXERCISE_ID)


def ask_ollama(raw_prediction: Dict[str, Any]) -> Dict[str, str]:
    return shared_ask_ollama(raw_prediction, EXERCISE_ID)