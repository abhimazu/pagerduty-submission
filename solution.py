import os
import sys
import json
import traceback
import argparse
from collections import deque, defaultdict
from datetime import timedelta

import pandas as pd
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Error: Set OPENAI_API_KEY environment variable.", file=sys.stderr)

client = OpenAI(api_key=OPENAI_API_KEY)

# Caching filenames
CHANGE_NOISE_CACHE_FILE = "cache/change_noise_cache.json"
INCIDENT_NOISE_CACHE_FILE = "cache/incident_noise_cache.json"
CAUSALITY_CACHE_FILE = "cache/causality_cache.json"
COUNT_PAIRS_CACHE_FILE = "cache/raw_count_pairs_cache.json"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Correlate incidents to changes with LLM-based filtering"
    )
    parser.add_argument("--changes", required=True)
    parser.add_argument("--incidents", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-minutes", type=int, default=60)
    parser.add_argument("--model", default="gpt-4o-mini")
    return parser.parse_args()


def load_cache(path):
    """Load cache from a JSON file."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}
    except json.JSONDecodeError:
        print(f"Error loading cache from {path}. Cache will be reset.", file=sys.stderr)
        return {}


def save_cache(cache, path):
    """Save cache to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache to {path}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def classify_with_llm(items, prompt_template, cache_file, model):
    """
    Classify items using LLM and cache results.

    Args:
        items: List of items to classify.
        prompt_template: Template for the prompt.
        cache_file: Path to the cache file.
        model: Model name.

    Returns:
        Dictionary of classified items.

    1. Load cache from the specified file.
    2. Initialize an empty dictionary for results.
    3. Iterate over each item in the input list.
    4. Create a unique key for each item.
    5. Check if the key is already in the cache.
    6. If found, retrieve the label from the cache.
    7. If not found, create a prompt using the template.
    8. Call the LLM API to get the label.
    9. Handle any exceptions during the API call.
    10. Save the label to the cache.
    11. Store the label in the results dictionary.
    12. Save the updated cache to the file.
    13. Return the results dictionary.
    14. Handle any unexpected errors and print the stack trace.
    """
    try:
        cache = load_cache(cache_file)
        results = {}

        for item in items:
            if not isinstance(item, str):
                key = f"{item[0]} ||| {item[1]}"
            else:
                key = item
            if key in cache:
                results[item] = cache[key]
                continue

            prompt = prompt_template.format(item=item)

            try:
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                label = response.output_text
            except Exception as e:
                print(f"Error classifying '{item}': {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                label = None

            cache[key] = label
            results[item] = label

        save_cache(cache, cache_file)
        return results

    except Exception as e:
        print(f"Unexpected error in classify_with_llm: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def load_and_prepare(change_path, incident_path):
    """Load and prepare data from CSV files."""
    try:
        changes = pd.read_csv(change_path)
        changes["timestamp"] = pd.to_datetime(
            changes["timestamp"], format="%Y-%m-%d %I:%M:%S %p"
        )
        changes = changes[["account_id", "service_id", "title", "timestamp"]]

        incidents = pd.read_csv(incident_path)
        incidents["timestamp"] = pd.to_datetime(
            incidents["triggered_at"], format="%Y-%m-%d %I:%M:%S %p"
        )
        incidents = incidents[["account_id", "service_id", "title", "timestamp"]]

        return changes, incidents
    except Exception as e:
        print(f"Error loading and preparing data: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def raw_correlate(changes, incidents, window_minutes):
    """ "
    Correlate changes and incidents based on a time window."

    Args:
        changes: DataFrame of changes.
        incidents: DataFrame of incidents.
        window_minutes: Time window in minutes.

    Returns:
        Dictionary of correlated pairs and their counts.

    1. Group changes and incidents by account_id and service_id.
    2. Find common groups between incidents and changes.
    3. For each incident, find changes within the time window.
    4. Count unique change titles for each incident.
    5. Return a dictionary of correlated pairs and their counts.
    6. Handle exceptions and print error messages.

    """
    try:
        window = timedelta(minutes=window_minutes)
        results = defaultdict(int)

        grp_chg = changes.groupby(["account_id", "service_id"])
        grp_inc = incidents.groupby(["account_id", "service_id"])

        common = set(grp_chg.groups) & set(grp_inc.groups)
        for key in common:
            chg = grp_chg.get_group(key).sort_values("timestamp")
            inc = grp_inc.get_group(key).sort_values("timestamp")

            dq = deque()
            idx = 0
            n = len(chg)

            for _, irow in inc.iterrows():
                its = irow["timestamp"]
                while idx < n and chg.iloc[idx]["timestamp"] <= its:
                    dq.append((chg.iloc[idx]["timestamp"], chg.iloc[idx]["title"]))
                    idx += 1
                cutoff = its - window
                while dq and dq[0][0] < cutoff:
                    dq.popleft()

                unique_titles = {t for _, t in dq}
                for ctitle in unique_titles:
                    results[(irow["title"], ctitle)] += 1

        return results

    except Exception as e:
        print(f"Error in raw correlation: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def save_raw_results(results):
    """Save raw results to a JSON file."""
    try:
        serializable = {
            json.dumps(key, ensure_ascii=False): count for key, count in results.items()
        }
        with open(COUNT_PAIRS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving raw results: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def filter_noise(changes, incidents, model):
    """
    Filters noise from changes and incidents using LLM classification."

    Args:
        changes: DataFrame of changes.
        incidents: DataFrame of incidents.
        model: Model name.

    Returns:
        Tuple of filtered changes and incidents.

    1. Classify change titles as MEANINGFUL or NOISE.
    2. Filter changes based on classification.
    3. Classify incident titles as MEANINGFUL or NOISE.
    4. Filter incidents based on classification.
    5. Return filtered changes and incidents.
    6. Handle exceptions and print error messages.
    """
    try:
        # Separately classify change titles
        change_titles = set(changes["title"])

        change_prompt = (
            "Classify the following CHANGE log title as MEANINGFUL or NOISE if the change can cause any incident:\n\n"
            + "{item}\n\nReply with exactly MEANINGFUL or NOISE."
        )
        labeled_changes = classify_with_llm(
            change_titles, change_prompt, CHANGE_NOISE_CACHE_FILE, model
        )
        valid_changes = {t for t, lbl in labeled_changes.items() if lbl == "MEANINGFUL"}
        fchg = changes[changes["title"].isin(valid_changes)].copy()

        # Separately classify incident titles
        incident_titles = set(incidents["title"])
        incident_prompt = (
            "Classify the following INCIDENT log title as MEANINGFUL or NOISE based on meaning:\n\n"
            + "{item}\n\nReply with exactly MEANINGFUL or NOISE."
        )
        labeled_incidents = classify_with_llm(
            incident_titles, incident_prompt, INCIDENT_NOISE_CACHE_FILE, model
        )
        valid_incidents = {
            t for t, lbl in labeled_incidents.items() if lbl == "MEANINGFUL"
        }
        finc = incidents[incidents["title"].isin(valid_incidents)].copy()

        return fchg, finc

    except Exception as e:
        print(f"Error in noise filtering: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def filter_causality(raw_results, model):
    """
    Filter causality using LLM classification.

    Args:
        raw_results: Dictionary of raw results.
        model: Model name.

    Returns:
        Dictionary of filtered results.

    1. Create a prompt for causality classification.
    2. Classify pairs using the LLM.
    3. Filter pairs based on classification.
    4. Return filtered results.
    5. Handle exceptions and print error messages.

    """
    try:
        prompt = (
            "We have a system change: '{item[1]}' and an incident: '{item[0]}'.\n"
            + "Reply with CAUSAL if the change likely caused the incident, otherwise NOT_CAUSAL."
        )
        label_map = classify_with_llm(raw_results, prompt, CAUSALITY_CACHE_FILE, model)

        final = {
            pair: cnt
            for pair, cnt in raw_results.items()
            if label_map.get(pair) == "CAUSAL"
        }
        return final

    except Exception as e:
        print(f"Error in causality filtering: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def write_results(results, output_path):
    """Write results to a JSON file."""
    try:
        out = {f"{i} ||| {c}": cnt for (i, c), cnt in results.items()}
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print(f"Error writing results: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to run the correlation process."""

    # Parse command line arguments
    args = parse_args()

    # Load and prepare data
    changes, incidents = load_and_prepare(args.changes, args.incidents)

    # Noise filtering
    clean_changes, clean_incidents = filter_noise(changes, incidents, args.model)

    # Raw correlation
    raw = raw_correlate(clean_changes, clean_incidents, args.window_minutes)

    # Save raw results
    save_raw_results(raw)

    # Causality filtering
    causal = filter_causality(raw, args.model)

    # Output File
    write_results(causal, args.output)

    print(f"Done: wrote {len(causal)} causal pairs to {args.output}")


if __name__ == "__main__":
    main()
