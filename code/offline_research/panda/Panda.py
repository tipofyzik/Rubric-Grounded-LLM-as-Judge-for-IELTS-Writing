"""Compatibility facade combining PANDA generation and retrieval workflows."""

from offline_research.panda.InsightGenerator import PandaInsightGenerator
from offline_research.panda.RetrievalEvaluator import PandaRetrievalEvaluator


class PANDA(PandaInsightGenerator, PandaRetrievalEvaluator):
    """Expose the existing PANDA workflow through its original class name."""

    pass
