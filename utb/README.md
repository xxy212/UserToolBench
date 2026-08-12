# UTB Code Release

This repository contains the core implementation for generating and evaluating personalized multi-turn tool-use benchmark data.

## Structure

- `generation/`: multi-agent data generation pipeline. It includes user simulators, planner/tool/checker/agent roles, model handlers, tool schemas, checkpointing, and visualization helpers.
- `evaluation/wtb/`: benchmark evaluation code for tool-call prediction, argument checking, graph-based execution matching, and model response generation.
- `evaluation/stats/`: scripts for dataset and score aggregation.
- `evaluation/scripts/eval_script.txt`: example commands for evaluation.
- `data_processing/translate.py`: persona translation and preprocessing utility.
- `.env.example`: environment variable template. Fill in API endpoints and keys before running.

## Environment

Create a Python environment with the common dependencies used by the generation and evaluation scripts:

```bash
pip install openai python-dotenv tqdm
```

Additional providers may require their own SDKs or OpenAI-compatible endpoints. The code assumes API credentials are provided through environment variables or a local `.env` file based on `.env.example`.

## Data Generation

The main personalized multi-turn generator is:

```bash
cd generation
cp ../.env.example .env
python personalized_generate.py --persona-limit 10 --min-user-turns 45 --max-user-turns 55
```

The generator loads persona profiles from `data/zh_persona`, samples tool-use task types, runs user/planner/tool/checker/agent roles, and writes generated sessions, checkpoints, and model-call logs under `generation/result`.

For the original single-session generation entry point:

```bash
cd generation
python generate.py
```

## Evaluation

Prepare benchmark jsonl files under `evaluation/data`, then run:

```bash
cd evaluation
cp ../.env.example .env
python wtb/openfunctions_evaluation.py --model deepseek-chat --num-threads 8 --data-dir ./data --result-dir ./result
python -m wtb.eval_runner --model deepseek-chat --data-dir ./data --result-dir ./result --score-dir ./score
```

The evaluator reads generated benchmark entries, calls the selected model, compares predicted tool calls against the reference tool-call graph, checks argument correctness, and writes model outputs plus metric files.

## Statistics

Dataset statistics:

```bash
cd evaluation
python stats/stats_datasets.py --data-dir ./data --output-dir ./stats_output
```

Score statistics:

```bash
cd evaluation
python stats/stats_score.py --score-dir ./score --output-dir ./stats_output
```

## Notes

This release intentionally includes only core code and tool schemas. Raw persona files, generated datasets, checkpoints, score files, cached bytecode, local paths, and private credentials are excluded.
