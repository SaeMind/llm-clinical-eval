"""
run_evaluation.py
LLM Clinical Summarization Evaluation Harness
Author: Andrew Lee | UTHealth Houston SBMI

Orchestrates the full benchmark pipeline:
  1. Load evaluation cases (synthetic clinical documents)
  2. Run each model on each task via async API calls
  3. Score outputs with automated metrics + LLM-as-judge
  4. Aggregate results into structured benchmark JSON
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluators.model_runner import ModelRunner
from src.evaluators.judge_evaluator import JudgeEvaluator
from src.metrics.automated_metrics import AutomatedMetrics
from src.metrics.hallucination_detector import HallucinationDetector
from src.benchmarks.synthetic_cases import SyntheticCaseGenerator
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ClinicalSummarizationBenchmark:
    """End-to-end benchmark orchestrator."""

    def __init__(self, eval_config: dict, model_config: dict):
        self.eval_cfg   = eval_config
        self.model_cfg  = model_config
        self.runner     = ModelRunner(model_config)
        self.judge      = JudgeEvaluator(eval_config["judge_model"])
        self.auto_metrics = AutomatedMetrics()
        self.halluc_det   = HallucinationDetector()
        self.results: list[dict] = []

    async def run_model_on_case(
        self,
        model_id: str,
        case: dict,
        task_type: str,
        prompt_template: str,
    ) -> dict:
        """Run one model on one evaluation case and score it."""
        prompt = prompt_template.format(
            source_document=case["source_document"],
            task_instruction=self.eval_cfg["tasks"][task_type]["instruction"],
        )

        # 1. Get model output
        model_output = await self.runner.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=512,
            temperature=0.0,  # deterministic for benchmarking
        )

        # 2. Automated metrics vs. reference summary
        auto_scores = self.auto_metrics.score(
            hypothesis=model_output,
            reference=case["reference_summary"],
            source=case["source_document"],
        )

        # 3. Hallucination detection (MedNLI entailment)
        halluc_scores = self.halluc_det.detect(
            summary=model_output,
            source=case["source_document"],
        )

        # 4. LLM-as-judge scoring
        judge_scores = await self.judge.score(
            source=case["source_document"],
            summary=model_output,
            task_type=task_type,
        )

        return {
            "model_id":           model_id,
            "task_type":          task_type,
            "case_id":            case["case_id"],
            "disease_category":   case["disease_category"],
            "model_output":       model_output,
            "auto_metrics":       auto_scores,
            "hallucination":      halluc_scores,
            "judge_scores":       judge_scores,
            "timestamp":          datetime.utcnow().isoformat(),
        }

    async def run_benchmark(self, n_cases_per_task: int = 20) -> list[dict]:
        """Run full benchmark across all models and tasks."""
        # Generate synthetic cases
        generator = SyntheticCaseGenerator()
        all_cases: dict[str, list[dict]] = {}
        for task_type in self.eval_cfg["tasks"]:
            all_cases[task_type] = generator.generate(
                task_type=task_type,
                n=n_cases_per_task,
            )
            logger.info(f"Generated {n_cases_per_task} cases for task: {task_type}")

        # Load prompt templates
        prompt_dir = Path("data/prompts")
        prompt_templates = {
            task: (prompt_dir / f"{task}.txt").read_text()
            for task in self.eval_cfg["tasks"]
        }

        # Build all evaluation jobs
        jobs = []
        for model_id in self.model_cfg["models"]:
            for task_type, cases in all_cases.items():
                for case in cases:
                    jobs.append((model_id, case, task_type, prompt_templates[task_type]))

        logger.info(f"Total evaluation jobs: {len(jobs)} "
                    f"({len(self.model_cfg['models'])} models x "
                    f"{len(self.eval_cfg['tasks'])} tasks x {n_cases_per_task} cases)")

        # Execute with concurrency limit to respect API rate limits
        semaphore = asyncio.Semaphore(self.eval_cfg.get("max_concurrent", 5))

        async def run_with_semaphore(job):
            async with semaphore:
                return await self.run_model_on_case(*job)

        results = await asyncio.gather(
            *[run_with_semaphore(j) for j in jobs],
            return_exceptions=True,
        )

        # Filter out exceptions
        successful = [r for r in results if not isinstance(r, Exception)]
        failed     = [r for r in results if isinstance(r, Exception)]
        if failed:
            logger.warning(f"{len(failed)} evaluation jobs failed: {failed[:3]}")

        self.results = successful
        logger.info(f"Benchmark complete: {len(successful)} results")
        return successful

    def aggregate_results(self) -> dict:
        """Compute per-model and per-task aggregate statistics."""
        from collections import defaultdict
        import numpy as np

        agg: dict[str, Any] = {"by_model": {}, "by_task": {}, "by_model_task": {}}

        def mean_safe(vals):
            return round(float(np.mean(vals)), 4) if vals else None

        for model_id in {r["model_id"] for r in self.results}:
            model_results = [r for r in self.results if r["model_id"] == model_id]
            agg["by_model"][model_id] = {
                "n_cases":              len(model_results),
                "mean_rouge1":          mean_safe([r["auto_metrics"]["rouge1"] for r in model_results]),
                "mean_bertscore_f1":    mean_safe([r["auto_metrics"]["bertscore_f1"] for r in model_results]),
                "mean_alignscore":      mean_safe([r["auto_metrics"].get("alignscore") for r in model_results if r["auto_metrics"].get("alignscore")]),
                "hallucination_rate":   mean_safe([r["hallucination"]["hallucination_rate"] for r in model_results]),
                "mean_halluc_count":    mean_safe([r["hallucination"]["hallucination_count"] for r in model_results]),
                "judge_factuality":     mean_safe([r["judge_scores"]["factuality"] for r in model_results]),
                "judge_completeness":   mean_safe([r["judge_scores"]["completeness"] for r in model_results]),
                "judge_safety_pass_rate": mean_safe([r["judge_scores"]["safety_pass"] for r in model_results]),
            }

        return agg


async def main(args):
    eval_config  = load_config(args.config)
    model_config = load_config(args.models)

    benchmark = ClinicalSummarizationBenchmark(eval_config, model_config)
    results   = await benchmark.run_benchmark(n_cases_per_task=args.n_cases)
    agg       = benchmark.aggregate_results()

    output = {
        "benchmark_metadata": {
            "run_timestamp":  datetime.utcnow().isoformat(),
            "n_models":       len(model_config["models"]),
            "n_tasks":        len(eval_config["tasks"]),
            "n_cases_per_task": args.n_cases,
            "judge_model":    eval_config["judge_model"],
        },
        "aggregate_results": agg,
        "raw_results":        results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {args.output}")

    # Print leaderboard
    print("\n=== BENCHMARK LEADERBOARD ===")
    print(f"{'Model':<45} {'ROUGE-1':>8} {'BERTScore':>10} {'Halluc%':>8} {'Judge Factuality':>17} {'Safety%':>8}")
    print("-" * 100)
    for model_id, metrics in sorted(
        agg["by_model"].items(),
        key=lambda x: -(x[1].get("judge_factuality") or 0)
    ):
        print(f"{model_id:<45} "
              f"{(metrics.get('mean_rouge1') or 0)*100:>7.1f}% "
              f"{(metrics.get('mean_bertscore_f1') or 0)*100:>9.1f}% "
              f"{(metrics.get('hallucination_rate') or 0)*100:>7.1f}% "
              f"{(metrics.get('judge_factuality') or 0):>16.2f}/5 "
              f"{(metrics.get('judge_safety_pass_rate') or 0)*100:>7.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Clinical Summarization Benchmark")
    parser.add_argument("--config",   default="configs/eval_config.yaml")
    parser.add_argument("--models",   default="configs/models.yaml")
    parser.add_argument("--output",   default="reports/benchmark_results.json")
    parser.add_argument("--n-cases",  type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args))
