"""
automated_metrics.py
ROUGE, BERTScore, and AlignScore computation for clinical summarization evaluation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AutomatedMetrics:
    """Wrapper for automated NLP evaluation metrics."""

    def __init__(self):
        self._rouge_scorer = None
        self._bert_scorer  = None
        self._align_scorer = None
        self._initialized  = False

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            from rouge_score import rouge_scorer
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], use_stemmer=True
            )
        except ImportError:
            logger.warning("rouge-score not installed. ROUGE metrics unavailable.")

        try:
            import bert_score
            self._bert_scorer = bert_score
        except ImportError:
            logger.warning("bert-score not installed. BERTScore unavailable.")

        try:
            from alignscore import AlignScore
            self._align_scorer = AlignScore(
                model="roberta-base",
                batch_size=8,
                device="cpu",
                evaluation_mode="nli_sp",
            )
        except ImportError:
            logger.warning("alignscore not installed. AlignScore unavailable.")

        self._initialized = True

    def score(self, hypothesis: str, reference: str, source: str) -> dict:
        """Compute all available automated metrics for a single summary."""
        self._lazy_init()
        results = {}

        # ROUGE
        if self._rouge_scorer:
            try:
                rouge_scores = self._rouge_scorer.score(reference, hypothesis)
                results["rouge1"]  = round(rouge_scores["rouge1"].fmeasure, 4)
                results["rouge2"]  = round(rouge_scores["rouge2"].fmeasure, 4)
                results["rougeL"]  = round(rouge_scores["rougeL"].fmeasure, 4)
            except Exception as e:
                logger.warning(f"ROUGE scoring failed: {e}")
                results.update({"rouge1": None, "rouge2": None, "rougeL": None})
        else:
            results.update({"rouge1": None, "rouge2": None, "rougeL": None})

        # BERTScore (F1)
        if self._bert_scorer:
            try:
                P, R, F1 = self._bert_scorer.score(
                    [hypothesis], [reference],
                    lang="en",
                    model_type="microsoft/deberta-xlarge-mnli",
                    verbose=False,
                )
                results["bertscore_p"]  = round(float(P[0]), 4)
                results["bertscore_r"]  = round(float(R[0]), 4)
                results["bertscore_f1"] = round(float(F1[0]), 4)
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")
                results.update({"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None})
        else:
            results.update({"bertscore_p": None, "bertscore_r": None, "bertscore_f1": None})

        # AlignScore (source-summary factual consistency)
        if self._align_scorer:
            try:
                align = self._align_scorer.score(
                    contexts=[source], claims=[hypothesis]
                )
                results["alignscore"] = round(float(align[0]), 4)
            except Exception as e:
                logger.warning(f"AlignScore failed: {e}")
                results["alignscore"] = None
        else:
            results["alignscore"] = None

        return results
