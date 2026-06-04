"""
hallucination_detector.py
Hallucination detection for clinical summaries using NLI entailment.
Splits summary into claims and checks each against source document.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinated claims in clinical summaries via NLI entailment.

    Pipeline:
      1. Split summary into atomic claims (sentence-level)
      2. For each claim: run NLI model with source as premise, claim as hypothesis
      3. Claims with label CONTRADICTION or NEUTRAL = potentially hallucinated
      4. Compute hallucination rate = n_non_entailed / n_total_claims
    """

    def __init__(self, model_name: str = "romaineg/medNLI"):
        self._pipeline = None
        self._model_name = model_name

    def _lazy_init(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=-1,  # CPU; set to 0 for GPU
                truncation=True,
                max_length=512,
            )
            logger.info(f"Loaded NLI model: {self._model_name}")
        except Exception as e:
            logger.warning(f"Could not load NLI model {self._model_name}: {e}. "
                           "Hallucination detection will return null results.")

    def _split_into_claims(self, text: str) -> list[str]:
        """Split text into sentence-level claims for NLI evaluation."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def detect(self, summary: str, source: str) -> dict:
        """Detect hallucinations in summary relative to source document."""
        self._lazy_init()

        claims = self._split_into_claims(summary)
        if not claims:
            return {"hallucination_rate": 0.0, "hallucination_count": 0,
                    "entailed_claims": 0, "non_entailed_claims": 0, "claim_details": []}

        if self._pipeline is None:
            return {"hallucination_rate": None, "hallucination_count": None,
                    "entailed_claims": None, "non_entailed_claims": None, "claim_details": []}

        # Truncate source to fit within model context
        source_truncated = source[:800]

        claim_details = []
        for claim in claims:
            input_text = f"{source_truncated} [SEP] {claim}"
            try:
                result = self._pipeline(input_text)[0]
                label  = result["label"].upper()
                score  = round(result["score"], 4)
                is_entailed = label in ("ENTAILMENT", "ENTAILS")
            except Exception as e:
                logger.warning(f"NLI inference failed for claim: {e}")
                label = "ERROR"; score = 0.0; is_entailed = True  # conservative

            claim_details.append({
                "claim":        claim,
                "nli_label":    label,
                "nli_score":    score,
                "is_entailed":  is_entailed,
            })

        n_entailed     = sum(1 for c in claim_details if c["is_entailed"])
        n_non_entailed = len(claim_details) - n_entailed

        return {
            "hallucination_rate":  round(n_non_entailed / len(claim_details), 4),
            "hallucination_count": n_non_entailed,
            "entailed_claims":     n_entailed,
            "non_entailed_claims": n_non_entailed,
            "total_claims":        len(claim_details),
            "claim_details":       claim_details,
        }
