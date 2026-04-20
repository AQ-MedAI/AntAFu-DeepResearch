# Main Implementation Code of WebClipper 

This repository contains a two-stage pipeline for building and refining trajectories from dialogue data. The final output is a rewritten trajectory suitable for training models.

## Overview

The pipeline consists of two scripts that must be run in order:

1. `state_graph_build.py`  
2. `mine_dag_and_message_refine.py`

The first script builds a state graph from raw conversation and saves the graph results. The second script takes these graph results, mines the minimal action subgraph (DAG), and refines the assistant messages to produce a compact, coherent trajectory.

---

## Environment Setup

1. Create and activate your Python environment.
2. Install dependencies, for example:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with content similar to:

   ```env
   # For state_graph_build.py
   EXTRACTOR_API_KEY=your_extractor_api_key
   EXTRACTOR_BASE_URL=your_extractor_base_url

   # For mine_dag_and_message_refine.py
   REWRITER_API_KEY=your_rewriter_api_key
   REWRITER_BASE_URL=your_rewriter_base_url

   # PPL model for candidate selection
   PPL_MODEL_PATH=your_ppl_model_path
   ```

Make sure the keys and URLs correspond to valid model endpoints that support the OpenAI-compatible chat API.

---

## 1. Build State Graph

Script: `state_graph_build.py`

This script reads raw conversation data and constructs a state graph containing information nodes, action nodes, and edges. It outputs a JSONL file where each line includes:

- `raw_message`: original message list
- `state_graph_result`: serialized state graph

### Example usage

```bash
python state_graph_build.py \
  --input /path/to/raw_conversations.jsonl \
  --output /path/to/state_graph_result.jsonl
```

The output file from this step will be used as the input to the second script.

---

## 2. Mine DAG and Refine Messages

Script: `mine_dag_and_message_refine.py`

This script takes the state graph results, performs:

1. Majority voting over repeated samples (if present) to get a stable minimal action subgraph.
2. Mining of the minimal action DAG leading to the final answer.
3. LLM-based refinement of the assistant’s internal thinking and responses, with optional PPL-based candidate selection.

The output is a JSONL file where each line typically contains:

- `messages`: the refined, pruned dialogue trajectory
- Additional metadata such as perplexity scores (`max_ppl`, `avg_ppl`, etc.)

### Example usage

```bash
python mine_dag_and_message_refine.py \
  --input /path/to/state_graph_result.jsonl \
  --output /path/to/refined_trajectory.jsonl
```

---

## End-to-End Flow

1. Prepare raw conversation data as JSONL.
2. Run `state_graph_build.py` to generate the state graph file.
3. Use that state graph file as the input to `mine_dag_and_message_refine.py`.
4. The final output (`refined_trajectory.jsonl`) can be used as training data, containing compact and refined trajectories.