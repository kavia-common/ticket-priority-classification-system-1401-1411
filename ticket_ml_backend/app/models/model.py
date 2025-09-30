"""
Machine learning model utilities for ticket classification.

This module provides a simple text classification model for classifying tickets
into priority labels: 'critical', 'major', or 'minor'.

The model is implemented using scikit-learn with a TfidfVectorizer and a
LogisticRegression classifier inside a Pipeline. It supports initial training
on sample data and incremental re-training when new labeled tickets are provided.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import threading

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DEFAULT_LABELS = ["critical", "major", "minor"]


@dataclass
class TrainingSample:
    """Represents a single training sample with text and label."""
    text: str
    label: str


class TicketPriorityModel:
    """
    Encapsulates a simple ML model for ticket priority classification.

    The model is a scikit-learn pipeline:
      - TfidfVectorizer for text feature extraction
      - LogisticRegression for multi-class classification

    Thread-safety:
      - Access to the underlying sklearn model is protected by a re-entrant lock
        to avoid race conditions during predict/train calls.
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self.labels: List[str] = labels or list(DEFAULT_LABELS)
        self._lock = threading.RLock()
        # Initialize with a basic pipeline
        self._pipeline: Pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, multi_class="auto")),
            ]
        )
        # Track whether the model has been fitted
        self._fitted: bool = False

    def _default_training_data(self) -> List[TrainingSample]:
        """
        Provide a minimal set of bootstrap training data for initial fit.
        In production, this could load from a persistent store.
        """
        samples = [
            # Critical
            TrainingSample("system down outage production not responding urgent asap", "critical"),
            TrainingSample("security breach data leak immediate action required", "critical"),
            TrainingSample("database unreachable 500 error critical incident", "critical"),
            # Major
            TrainingSample("payment service failing intermittently needs attention", "major"),
            TrainingSample("performance degradation on dashboard slow loading", "major"),
            TrainingSample("email notifications not sent for some users", "major"),
            # Minor
            TrainingSample("UI alignment issue on settings page low impact", "minor"),
            TrainingSample("typo in help text on login screen cosmetic", "minor"),
            TrainingSample("feature request add dark mode when possible", "minor"),
        ]
        return samples

    # PUBLIC_INTERFACE
    def ensure_fitted(self) -> None:
        """Ensure the model is fitted at least once using default training data."""
        with self._lock:
            if self._fitted:
                return
            samples = self._default_training_data()
            X = [s.text for s in samples]
            y = [s.label for s in samples]
            self._pipeline.fit(X, y)
            self._fitted = True

    # PUBLIC_INTERFACE
    def predict(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Predict the priority label for the given text.

        Returns:
          - predicted label
          - dict of class probabilities per label
        """
        self.ensure_fitted()
        with self._lock:
            proba = self._pipeline.predict_proba([text])[0]
            classes = list(self._pipeline.classes_)
            # Map probabilities to the known labels. If model classes differ,
            # still return a full mapping for transparency.
            prob_map: Dict[str, float] = {cls: float(p) for cls, p in zip(classes, proba)}
            # Best label
            best_idx = int(proba.argmax())
            label = classes[best_idx]
            # Ensure all default labels present in map (possibly with 0.0 if absent)
            for default in self.labels:
                if default not in prob_map:
                    prob_map[default] = 0.0
            return label, prob_map

    # PUBLIC_INTERFACE
    def train(self, samples: List[TrainingSample]) -> None:
        """
        Train (fit) the model from scratch using provided samples.

        In a production-ready system, you may want to support incremental learning
        or cumulative fitting with partial_fit algorithms. For simplicity, we
        re-fit the pipeline.
        """
        if not samples:
            # No-op if no data provided
            return
        with self._lock:
            X = [s.text for s in samples]
            y = [s.label for s in samples]
            self._pipeline.fit(X, y)
            self._fitted = True

    # PUBLIC_INTERFACE
    def retrain_with_augmented(self, extra_samples: List[TrainingSample]) -> None:
        """
        Retrain the model combining default training data with extra labeled samples.
        """
        base = self._default_training_data()
        all_samples = base + (extra_samples or [])
        self.train(all_samples)
