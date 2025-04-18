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
CHANGE_NOISE_CACHE_FILE = "change_noise_cache.json"
INCIDENT_NOISE_CACHE_FILE = "incident_noise_cache.json"
CAUSALITY_CACHE_FILE = "causality_cache.json"
COUNT_PAIRS_CACHE_FILE = "raw_count_pairs_cache.json"


def parse_args():
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
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def classify_with_llm(items, prompt_template, cache_file, model):
    cache = load_cache(cache_file)
    results = {}

    for item in items:
        print(f"Processing item: {item}")
        if not isinstance(item, str):
            key = f"{item[0]} ||| {item[1]}"
        else:
            key = item
        print(f"Key: {key}")
        print(f"Cache Keys: {cache.keys()}")
        if key in cache:
            results[item] = cache[key]
            continue

        prompt = prompt_template.format(item=item)

        try:
            response = client.responses.create(
                model=model, input=[{"role": "user", "content": prompt}], temperature=0
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


def load_and_prepare(change_path, incident_path):
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


def raw_correlate(changes, incidents, window_minutes):
    window = timedelta(minutes=window_minutes)
    results = defaultdict(int)

    grp_chg = changes.groupby(["account_id", "service_id"])
    print(f"Changes: {grp_chg.groups.keys()}")
    grp_inc = incidents.groupby(["account_id", "service_id"])
    print(f"Incidents: {grp_inc.groups.keys()}")

    common = set(grp_chg.groups) & set(grp_inc.groups)
    print(f"Common: {common}")
    for key in common:
        print(f"Processing {key}")

        chg = grp_chg.get_group(key).sort_values("timestamp")
        print(f"Changes: {chg}")
        inc = grp_inc.get_group(key).sort_values("timestamp")
        print(f"Incidents: {inc}")

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


def save_raw_results(results):
    serializable = {
        json.dumps(key, ensure_ascii=False): count for key, count in results.items()
    }
    # write out to JSON
    with open(COUNT_PAIRS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def get_count_dict_from_raw(results):
    # Convert keys into strings with || separator
    count_dict = {f"{i} ||| {c}": cnt for (i, c), cnt in results.items()}
    return count_dict


def filter_noise(changes, incidents, model):
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
    valid_incidents = {t for t, lbl in labeled_incidents.items() if lbl == "MEANINGFUL"}
    finc = incidents[incidents["title"].isin(valid_incidents)].copy()

    return fchg, finc


def filter_causality(raw_results, model):
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


def write_results(results, output_path):
    out = {f"{i} ||| {c}": cnt for (i, c), cnt in results.items()}
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    args = parse_args()

    changes, incidents = load_and_prepare(args.changes, args.incidents)

    # 1. Noise filtering
    clean_changes, clean_incidents = filter_noise(changes, incidents, args.model)

    # 2. Raw correlation
    raw = raw_correlate(clean_changes, clean_incidents, args.window_minutes)

    save_raw_results(raw)

    # 3. Causality filtering 
    causal = filter_causality(raw, args.model)

    # 4. Output
    write_results(causal, args.output)

    print(f"Done: wrote {len(causal)} causal pairs to {args.output}")


if __name__ == "__main__":
    main()
