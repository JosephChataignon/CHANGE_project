# -*- coding: utf-8 -*-
"""
Quantitative evaluation of FRAG semantic search on synthetic data.
(Evaluation round 2, step II.B)

For each synthetic (query, relevant_sentence) pair produced by synthetic_data_gen.py,
submit the query to the FRAG search API and check whether the original chunk is
among the retrieved sources, and at which rank.

Because the documents returned by FRAG may be chunked differently from how we chunked
them when generating the synthetic data, we match by TEXT OVERLAP (word n-gram
containment) between each retrieved source and the original `relevant_sentence`,
rather than by document reference.

Outputs, written into output_dir (use a different output_dir for each embedding model
so runs of fine-tuned and non-fine-tuned models can be compared):
  - results.csv .............. one row per query: per-rank overlaps, best overlap, best rank
  - overlap_distribution.png .. histogram of best overlap per query, used to choose the
                               "same document" threshold
  - summary.json ............. top-1/5/10 hit rates at all candidate thresholds,
                               plus run configuration
  - report.txt ................ human-readable report: the hit-rate table at all
                               candidate thresholds

Usage:
    python benchmarking.py [output_dir] [n_samples]

Arguments:
    output_dir : directory where output files are written (default: current directory)
    n_samples  : number of synthetic question-document pairs to evaluate (default: all)
"""

import sys, os, re, csv, json, logging
from datetime import datetime

import requests
import matplotlib
matplotlib.use("Agg")  # headless backend, we only save plots to file
import matplotlib.pyplot as plt
from tqdm import tqdm

# ############################ Configuration ############################

# Placeholder: file produced by synthetic_data_gen.py, with columns 'query' and
# 'relevant_sentence'. Change to the actual path once the synthetic data is generated.
SYNTHETIC_DATA_PATH = "/mnt/wbkolleg_dh_1/Projekt_Change_LLM/Eduscience_data/SyntheticQA/synthetic_data_qwen3-8-oct14.csv"

TOP_KS = (1, 5, 10)   # ranks at which we measure hits
N_RESULTS = max(TOP_KS)  # number of sources to request (>= max(TOP_KS))

NGRAM_SIZE = 5                          # word n-gram size for text overlap
CANDIDATE_THRESHOLDS = [0.1 * i for i in range(1, 10)]  # hit rates reported for each
DEFAULT_THRESHOLD = 0.5                 # overlap above which a source counts as same document

# ############################ Text overlap #############################

def tokenize(text):
    return re.findall(r"\w+", text.lower(), re.UNICODE)

def word_ngrams(tokens):
    return {tuple(tokens[i:i + NGRAM_SIZE]) for i in range(len(tokens) - NGRAM_SIZE + 1)}

def overlap_score(original, retrieved):
    """Text overlap between the original chunk and a retrieved source, in [0, 1].

    Word n-gram containment, taking the max of both directions: chunking boundaries
    differ between synthetic data generation and FRAG ingestion, so the retrieved
    chunk may contain only part of the original chunk, or vice versa.
    """
    grams_orig = word_ngrams(tokenize(original))
    grams_retr = word_ngrams(tokenize(retrieved))
    if not grams_orig or not grams_retr:
        return 0.0
    intersection = len(grams_orig & grams_retr)
    return max(intersection / len(grams_orig), 
               intersection / len(grams_retr))

# ############################ Evaluation ############################

def evaluate_pair(query, relevant_sentence):
    """Run one synthetic pair through the search API. Returns per-rank overlaps."""
    response = requests.post("http://change.dh.unibe.ch/api/search/", 
                             json={"query": query, "n_results": N_RESULTS},
                             timeout=120).json()
    retrieved_texts = [doc["content"] for doc in response["documents"]]
    return [overlap_score(relevant_sentence, text) for text in retrieved_texts]

def compute_hit_rates(results, threshold):
    """Fraction of queries whose original chunk is found at rank <= k."""
    n = len(results)
    return {f"top_{k}": sum(any(o >= threshold for o in overlaps[:k])
                           for overlaps in results) / n
            for k in TOP_KS}

# ############################ Outputs ############################

def save_results_csv(results, queries, sentences, output_path):
    max_rank = max(len(o) for o in results)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "relevant_sentence",
                         "best_overlap", "best_rank"]
                        + [f"overlap_rank_{i + 1}" for i in range(max_rank)])
        for query, sentence, overlaps in zip(queries, sentences, results):
            best = max(overlaps)
            writer.writerow([query, sentence, best, overlaps.index(best) + 1]
                            + overlaps + [None] * (max_rank - len(overlaps)))

def make_overlap_distribution_plot(results, output_path):
    """Histogram of best overlap per query. Typically bimodal: high overlaps are
    true matches, low overlaps are misses. The valley between the modes is a good
    choice for the 'same document' threshold."""
    best_overlaps = [max(o) for o in results]
    plt.figure(figsize=(8, 5))
    plt.hist(best_overlaps, bins=20, range=(0.0, 1.0), edgecolor="black")
    plt.axvline(DEFAULT_THRESHOLD, color="red", linestyle="--",
                label=f"default threshold = {DEFAULT_THRESHOLD}")
    plt.yscale("log")
    plt.xlabel("Best overlap (word n-gram containment)")
    plt.ylabel("Number of queries (log scale)")
    plt.title(f"Retrieved vs original chunk overlap\nn = {len(results)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# ############################ Main ############################

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
    ## Load arguments
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None

    ## Load the synthetic data
    with open(SYNTHETIC_DATA_PATH, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["query", "relevant_sentence"], \
            f"Unexpected columns in {SYNTHETIC_DATA_PATH}: {reader.fieldnames}"
        pairs = [(row["query"], row["relevant_sentence"]) for row in reader]
    if n_samples: # limit to requested number of samples if specified
        pairs = pairs[:n_samples]
    logging.info(f"Loaded {len(pairs)} synthetic question-document pairs")

    ## Run the benchmark
    results, queries, sentences = [], [], []
    n_errors = 0
    for query, sentence in tqdm(pairs, desc="Evaluating queries"):
        try:
            results.append(evaluate_pair(query, sentence))
        except Exception as e:  # bad response for this query: log, skip, keep going
            n_errors += 1
            logging.warning(f"Skipping query '{query[:50]}...': {e}")
        else:
            queries.append(query)
            sentences.append(sentence)
    logging.info(f"Evaluated {len(results)} queries ({n_errors} skipped due to errors)")

    ## Per-query results
    results_path = os.path.join(output_dir, "results.csv")
    save_results_csv(results, queries, sentences, results_path)
    logging.info(f"Per-query results saved to: {results_path}")

    ## Overlap distribution plot
    plot_path = os.path.join(output_dir, "overlap_distribution.png")
    make_overlap_distribution_plot(results, plot_path)
    logging.info(f"Overlap distribution plot saved to: {plot_path}")

    ## Hit rates at every candidate threshold, to help choose the threshold
    threshold_table = {str(t): compute_hit_rates(results, t)
                       for t in CANDIDATE_THRESHOLDS}

    ## Summary
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_queries_evaluated": len(results),
        "n_queries_failed": n_errors,
        "ngram_size": NGRAM_SIZE,
        "overlap_metric": "max directional word n-gram containment",
        "hit_rates_by_threshold": threshold_table,
        "hit_rates_default_threshold": {"threshold": DEFAULT_THRESHOLD,
                                        **compute_hit_rates(results, DEFAULT_THRESHOLD)},
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"Summary saved to: {summary_path}")

    ## Human-readable report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("FRAG search benchmark\n")
        f.write(f"=====================\n\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Queries evaluated: {len(results)} ({n_errors} skipped due to errors)\n\n")
        f.write("Hit rates by overlap threshold (use this to choose the threshold "
                "for 'same document'):\n")
        f.write(f"{'threshold':>10} | {'top-1':>7} {'top-5':>7} {'top-10':>7}\n")
        for t, hit_rates in threshold_table.items():
            f.write(f"{float(t):>10.1f} | {hit_rates['top_1']:>7.3f} "
                    f"{hit_rates['top_5']:>7.3f} {hit_rates['top_10']:>7.3f}\n")
        f.write(f"\nAt default threshold {DEFAULT_THRESHOLD}:\n")
        for k in TOP_KS:
            f.write(f"  top_{k}: {summary['hit_rates_default_threshold'][f'top_{k}']:.3f}\n")
    logging.info(f"Report saved to: {report_path}")
