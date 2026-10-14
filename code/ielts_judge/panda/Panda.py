"""Compatibility facade combining PANDA generation and retrieval workflows."""

from .InsightGenerator import PandaInsightGenerator
from .RetrievalEvaluator import PandaRetrievalEvaluator


class PANDA(PandaInsightGenerator, PandaRetrievalEvaluator):
    """Expose the existing PANDA workflow through its original class name."""

    pass
