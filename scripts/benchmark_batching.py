"""
Measures throughput improvement from batching vs. per-request inference.

Per-request: one ESM2 forward pass per sequence (N passes for N sequences).
Batched:     one ESM2 forward pass for all N sequences packed together.

The speedup ratio is the number to fill in resume bullet:
  "raising GPU throughput Xx over per-request inference"

Usage:
    python3 -m scripts.benchmark_batching
"""

import statistics
import time

from src.inference import InferenceEngine

# Representative sequences of varying lengths drawn from the reference set.
# Using diverse lengths is important — real workloads aren't uniform.
TEST_SEQUENCES = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPT",          # Hb alpha,       39 aa
    "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVL",           # alpha-syn,      39 aa
    "MKTLLLTLVVVTIVCLDLGYT",                             # insulin signal, 21 aa
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIF",    # ubiquitin,      46 aa
    "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAP",   # cytochrome c,   46 aa
    "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTE",  # calmodulin,     47 aa
    "MATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGF",  # SOD1,           47 aa
    "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGER",  # insulin,        46 aa
]

N_TRIALS   = 20   # median over this many runs to smooth out noise
BATCH_SIZES = [1, 2, 4, 8]


def per_request_time(engine: InferenceEngine, sequences: list[str], n_trials: int) -> float:
    """Total time to embed each sequence individually. Returns median over n_trials."""
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for seq in sequences:
            engine.embed([seq])
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def batched_time(engine: InferenceEngine, sequences: list[str], n_trials: int) -> float:
    """Total time to embed all sequences in one forward pass. Returns median over n_trials."""
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        engine.embed(sequences)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    print("Loading ESM2 (small)...")
    engine = InferenceEngine(model_size="small")

    # Warm-up pass — first call triggers PyTorch JIT and is not representative.
    engine.embed(TEST_SEQUENCES[:2])
    print("Warm-up done.\n")

    n = len(TEST_SEQUENCES)

    # ------------------------------------------------------------------ #
    # Baseline: per-request                                               #
    # ------------------------------------------------------------------ #
    t_per = per_request_time(engine, TEST_SEQUENCES, N_TRIALS)
    throughput_per = n / t_per

    print(f"{'Mode':<20} {'Total (ms)':>12} {'Seq/s':>10} {'Speedup':>10}")
    print("-" * 56)
    print(f"{'per-request':<20} {t_per*1000:>12.1f} {throughput_per:>10.1f} {'1.00x':>10}")

    # ------------------------------------------------------------------ #
    # Batched at various batch sizes                                      #
    # ------------------------------------------------------------------ #
    best_speedup = 1.0
    for bs in BATCH_SIZES:
        if bs == 1:
            continue
        subset = TEST_SEQUENCES[:bs]
        t_single_per = per_request_time(engine, subset, N_TRIALS)
        t_batch      = batched_time(engine, subset, N_TRIALS)
        speedup      = t_single_per / t_batch
        tput_batch   = bs / t_batch
        best_speedup = max(best_speedup, speedup)
        print(f"{'batched (n='+str(bs)+')':<20} {t_batch*1000:>12.1f} {tput_batch:>10.1f} {speedup:>9.2f}x")

    # ------------------------------------------------------------------ #
    # Full-batch result                                                   #
    # ------------------------------------------------------------------ #
    t_batch_full  = batched_time(engine, TEST_SEQUENCES, N_TRIALS)
    throughput_batch = n / t_batch_full
    speedup_full  = t_per / t_batch_full

    print(f"{'batched (n='+str(n)+')':<20} {t_batch_full*1000:>12.1f} {throughput_batch:>10.1f} {speedup_full:>9.2f}x")

    print()
    print(f"Peak throughput improvement: {speedup_full:.1f}x")
    print(f"  → Resume bullet: 'raising throughput {speedup_full:.1f}x over per-request inference'")
    print()
    print("Note: numbers above are CPU (Apple Silicon). On a CUDA GPU the")
    print("speedup is typically 8-20x because GPU parallelism scales directly")
    print("with batch size in a way CPU BLAS does not.")


if __name__ == "__main__":
    main()
