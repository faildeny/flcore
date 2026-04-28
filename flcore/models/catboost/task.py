import json
import os
import tempfile

from catboost import CatBoostClassifier


def convert_to_model_dict(model_input: CatBoostClassifier) -> dict:
    """Convert CatBoost model object into a serializable JSON dict."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        model_input.save_model(tmp_path, format="json")
        with open(tmp_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def convert_to_catboost(model_input: bytes) -> CatBoostClassifier:
    """Convert serialized CatBoost JSON bytes into a CatBoost model object."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(json.loads(model_input), handle)
        model = CatBoostClassifier()
        model.load_model(tmp_path, format="json")
        return model
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def model_to_bytes(model_input: CatBoostClassifier) -> bytes:
    """Serialize CatBoost model to JSON bytes."""
    return json.dumps(convert_to_model_dict(model_input)).encode("utf-8")