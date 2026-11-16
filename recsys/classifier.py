"""Loading the trained action classifier and using it. Training lives in pipeline/build_app_model.py."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from .cleaning import clean


def model_text(negative: str, positive: str) -> str:
    """The text the model reads. The positive side is repeated because it carries more signal.
    Training and the app both call this, so they cannot drift apart."""
    return f"{negative} . {positive} . {positive}"


@dataclass
class ActionModel:
    vectorizer: object
    models: list
    thresholds: dict
    names: list
    typo_map: dict
    version: str

    def predict(self, negative: str, positive: str) -> dict:
        """Score (0 to 1) for each action, from the raw text typed by the user."""
        x = self.vectorizer.transform([model_text(clean(negative, self.typo_map), clean(positive, self.typo_map))])
        return {name: float(m.predict_proba(x)[0, 1]) for name, m in zip(self.names, self.models, strict=True)}

    def ranked(self, scores: dict) -> list:
        """(action, score) with the actions closest to their own threshold first."""
        return sorted(scores.items(), key=lambda kv: -(kv[1] / self.thresholds[kv[0]]))


def load_model(model_path: Path, typo_path: Path) -> ActionModel:
    saved = joblib.load(model_path)
    return ActionModel(
        vectorizer=saved["tf"],
        models=saved["models"],
        names=saved["names"],
        thresholds={a: float(t) for a, t in zip(saved["names"], saved["th"], strict=True)},
        typo_map=json.loads(Path(typo_path).read_text(encoding="utf-8")),
        version=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:10],
    )
