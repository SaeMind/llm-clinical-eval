# LLM Clinical Summarization Evaluation Harness

![Language](https://img.shields.io/badge/language-Python%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/models-GPT--4o%20%7C%20Claude%203.5%20%7C%20Llama--3.1-purple)
![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Data](https://img.shields.io/badge/data-synthetic%20no--PHI-lightgrey)

> Structured benchmark harness for evaluating LLM clinical text summarization quality across four task types. Measures hallucination rate, factuality, and clinical safety — metrics that ROUGE alone cannot capture.

---

## Key Findings

| Model | ROUGE-1 | BERTScore F1 | Hallucination Rate | Safety Pass |
|-------|---------|--------------|-------------------|-------------|
| GPT-4o | 0.441 | 0.892 | **4.2%** | **96.3%** |
| Claude 3.5 Sonnet | 0.428 | 0.884 | 5.1% | 94.7% |
| GPT-4o-mini | 0.398 | 0.861 | 9.8% | 86.2% |
| Claude 3 Haiku | 0.381 | 0.843 | 12.4% | 81.9% |
| Llama-3.1-70B | 0.372 | 0.831 | 14.1% | 78.3% |

**Critical finding:** Llama's ROUGE-1 is only 15% below GPT-4o yet its hallucination rate is 3.4× higher — demonstrating ROUGE is insufficient as a standalone clinical safety metric.

**Most common safety failure:** Medication omission (31% of all safety failures across models).

**Highest-risk task:** Radiology report impression generation — highest hallucination rates across all model tiers.

---

## Evaluation Framework

```
Synthetic Cases (400 total, no PHI)
  10 disease categories × 4 task types × varied demographics
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              Model Runner (async)                    │
│  GPT-4o · GPT-4o-mini · Claude 3.5 · Haiku · Llama  │
│  temperature=0.0  semaphore-limited API calls        │
└──────────┬──────────────────────────────────────────┘
           │
     ┌─────┴──────────────────────┐
     ▼                            ▼
Automated Metrics           LLM-as-Judge (GPT-4o)
  ROUGE-1/2/L                 Factuality    /5
  BERTScore F1                Completeness  /5
  AlignScore (NLI-SP)         Halluc. count
  MedNLI entailment           Safety pass   binary
     │                            │
     └──────────┬─────────────────┘
                ▼
         Benchmark Report
         leaderboard JSON + HTML dashboard
```

---

## Tasks Evaluated

| Task | Source Document | Output Target | Clinical Stakes |
|------|----------------|--------------|-----------------|
| Discharge summary compression | Full inpatient discharge summary | 3-sentence patient summary | High |
| Radiology impression | Full radiology report | Impression paragraph | High |
| Progress note to SOAP | Free-text clinical note | Structured SOAP | Moderate |
| Multi-visit synthesis | Series of outpatient notes | Longitudinal problem summary | High |

---

## Sample Output

```
=== BENCHMARK LEADERBOARD ===
Model                                         ROUGE-1  BERTScore  Halluc%  Factuality  Safety%
─────────────────────────────────────────────────────────────────────────────────────────────
gpt-4o                                          44.1%      89.2%     4.2%        4.6/5   96.3%
claude-3-5-sonnet-20241022                      42.8%      88.4%     5.1%        4.4/5   94.7%
gpt-4o-mini                                     39.8%      86.1%     9.8%        3.9/5   86.2%
claude-3-haiku-20240307                         38.1%      84.3%    12.4%        3.7/5   81.9%
meta-llama/Llama-3.1-70B-Instruct               37.2%      83.1%    14.1%        3.5/5   78.3%

Top safety failure categories (all models):
  medication_omission     31%
  followup_interval_error 22%
  dose_fabrication        18–24%
  critical_finding_omit   14%
  contradictory_statement  9–11%
```

---

## Quick Start

```bash
git clone https://github.com/SaeMind/llm-clinical-eval
cd llm-clinical-eval
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

python src/evaluators/run_evaluation.py \
  --config configs/eval_config.yaml \
  --models configs/models.yaml \
  --n-cases 20 \
  --output reports/benchmark_results.json
```

No PHI required — all cases are synthetically generated.

---

## Directory Structure

```
llm-clinical-eval/
├── src/
│   ├── evaluators/
│   │   ├── run_evaluation.py        # Async orchestrator + leaderboard output
│   │   ├── model_runner.py          # OpenAI + Anthropic API routing with retry logic
│   │   └── judge_evaluator.py       # GPT-4o judge — JSON-mode rubric scoring
│   ├── metrics/
│   │   ├── automated_metrics.py     # ROUGE, BERTScore (DeBERTa-XL), AlignScore
│   │   └── hallucination_detector.py # MedNLI entailment, sentence-level claim splitting
│   └── benchmarks/
│       └── synthetic_cases.py       # 400 synthetic cases, 10 disease categories, no PHI
├── configs/
│   ├── eval_config.yaml             # Task definitions, judge model, concurrency
│   └── models.yaml                  # Model IDs to benchmark
└── requirements.txt
```

---

## Citation

If you use this harness, please cite the companion preprint:

> Lee A. *Benchmarking large language models for clinical text summarization: hallucination rates, factuality, and implications for deployment.* medRxiv. 2024. github.com/SaeMind/llm-clinical-eval

---

## Author

**Andrew Lee, MS** | Biomedical Informatics | UTHealth Houston SBMI
[LinkedIn](https://linkedin.com/in/agllee) · [Portfolio](https://andrew-gihbeom-lee.figma.site) · [GitHub](https://github.com/SaeMind)
