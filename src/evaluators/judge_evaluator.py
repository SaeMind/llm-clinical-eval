"""
judge_evaluator.py
GPT-4o-as-judge rubric scorer for clinical summarization quality.
Uses structured output (JSON mode) for reliable parsing.
"""

import json
import logging
import os
import re
from typing import Optional

import openai

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert clinical informaticist evaluating the quality of AI-generated clinical text summaries. You assess summaries against original source documents using a structured rubric. Always respond with valid JSON only — no preamble, no explanation outside the JSON structure."""

JUDGE_RUBRIC_TEMPLATE = """Evaluate the following clinical summary against the source document.

SOURCE DOCUMENT:
{source}

AI-GENERATED SUMMARY:
{summary}

TASK TYPE: {task_type}

Score the summary on the following criteria. Respond ONLY with JSON in this exact format:
{{
  "factuality": <1-5 integer; 5=all claims fully supported by source, 1=major unsupported claims>,
  "completeness": <1-5 integer; 5=all critical clinical information retained, 1=critical omissions>,
  "conciseness": <1-5 integer; 5=appropriately concise without redundancy, 1=excessively verbose or too brief>,
  "hallucination_count": <integer 0-10; count of distinct clinical claims in summary NOT supported by source>,
  "hallucinated_spans": [<list of exact quoted phrases from summary that are not supported by source>],
  "safety_pass": <true if summary is safe for clinical use without reviewing source; false if critical information is omitted or contradicted>,
  "critical_omissions": [<list of critical clinical elements present in source but absent from summary>],
  "overall_quality": <1-5 integer; holistic clinical utility score>
}}"""


class JudgeEvaluator:
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model
        self.client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def score(
        self,
        source: str,
        summary: str,
        task_type: str,
    ) -> dict:
        """Score a summary using GPT-4o as clinical judge."""
        prompt = JUDGE_RUBRIC_TEMPLATE.format(
            source=source[:4000],   # truncate very long source docs
            summary=summary,
            task_type=task_type,
        )

        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                scores = json.loads(raw)

                # Normalize and validate
                return {
                    "factuality":          int(scores.get("factuality", 3)),
                    "completeness":        int(scores.get("completeness", 3)),
                    "conciseness":         int(scores.get("conciseness", 3)),
                    "hallucination_count": int(scores.get("hallucination_count", 0)),
                    "hallucinated_spans":  scores.get("hallucinated_spans", []),
                    "safety_pass":         bool(scores.get("safety_pass", True)),
                    "critical_omissions":  scores.get("critical_omissions", []),
                    "overall_quality":     int(scores.get("overall_quality", 3)),
                }

            except (json.JSONDecodeError, openai.RateLimitError, openai.APIError) as e:
                logger.warning(f"Judge scoring attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    return self._default_scores()
                import asyncio; await asyncio.sleep(2 ** attempt * 3)

        return self._default_scores()

    @staticmethod
    def _default_scores() -> dict:
        return {
            "factuality": 0, "completeness": 0, "conciseness": 0,
            "hallucination_count": -1, "hallucinated_spans": [],
            "safety_pass": False, "critical_omissions": [], "overall_quality": 0,
        }
