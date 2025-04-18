# Correlating Change Events to Incident Events

This repository contains a Python-based solution to identify and count unique causal relationships between system change events and incident events within a configurable time window (default: 60 minutes).

## Table of Contents

- [Assumptions](#assumptions)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture & Flow](#architecture--flow)
- [Key Modules](#key-modules)
- [Next Steps & Improvements](#next-steps--improvements)

## Assumptions

- **Match Scope**: Only correlate events sharing the same `account_id` and `service_id`.
- **Time Window**: Inclusive 60 minutes prior to each incident timestamp (configurable).
- **Deduplication**: Change titles deduped by `(account_id, service_id, title)`—only the first occurrence per incident window counts.
- **Noise Filtering**: Titles classified as `NOISE` by the LLM are removed before correlation.
- **Output**: JSON mapping of `"incident_title ||| change_title" → count`; incidents with zero matches are omitted.

## Features

- **CSV Ingestion**: Reads and normalizes `change_events.csv` and `incident_events.csv`.
- **LLM-Based Noise Removal**: Filters out unimportant change and incident titles using OpenAI API.
- **Efficient Window Correlation**: Uses a deque to maintain a sliding window of relevant change events.
- **Causality Confirmation**: Verifies likely causality of each pair using an LLM.
- **JSON Output**: Writes final causal relationships to a JSON file.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"    # Linux/macOS
   set OPENAI_API_KEY="your_api_key_here"       # Windows
   ```

## Usage

```bash
python correlate.py \
  --changes change_events.csv \
  --incidents incident_events.csv \
  --output final_output.json \
  --window-minutes 60 \
  --model gpt-4.1-nano
```

- `--window-minutes`: Time window in minutes (default: 60).
- `--model`: OpenAI model to use for LLM-based filtering (default: `gpt-4o-mini`).

## Architecture & Flow

```mermaid
flowchart LR
    A[Start] --> B[Load & Parse CSVs]
    B --> C[Noise Filtering]
    C --> D[Raw Correlation (Sliding Window)]
    D --> E[Save Raw Results]
    E --> F[Causality Filtering via LLM]
    F --> G[Write Final JSON Output]
    G --> H[End]
```

## Key Functions

- **`load_and_prepare()`**: Ingests CSVs, parses timestamps, and selects necessary fields.
- **`filter_noise()`**: Calls LLM to understand each unique title within incidents, change records and removes `NOISE` titles, caching results to avoid repeat LLM calls.
- **`raw_correlate()`**: Implements the sliding window algorithm with a deque to correlate events within the given time period.
- **`filter_causality()`**: Uses LLM classification to confirm true causal pairs from the above correlation.
- **`write_results()`**: Serializes the final counts to JSON.
