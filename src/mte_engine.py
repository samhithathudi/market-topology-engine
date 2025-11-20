from typing import Dict, Any, List, Tuple

from .lob_generator import generate_lob_time_series
from .lob_shapes import (
    lob_to_pointcloud,
    pointcloud_topology_features,
    lob_shape_distance,
    topology_stress_score,
    detect_regime_shifts,
)


def run_mte_simulation(
    n_steps: int = 100,
    gap_threshold: float = 0.02,
    regime_k: float = 2.0,
) -> Dict[str, Any]:
    """
    Run a full Market-Topology Engine simulation on synthetic data.

    Returns a dict with:
      - series: list of LOB DataFrames
      - stress_scores: list[float]
      - threshold: float
      - shift_indices: list[int]
    """
    series = generate_lob_time_series(n_steps=n_steps)

    stress_scores: List[float] = []

    for i in range(len(series) - 1):
        lob_now = series[i]
        lob_next = series[i + 1]

        # shape deformation between t and t+1
        shape_dist = lob_shape_distance(lob_now, lob_next)

        # topology-like features at time t
        pc = lob_to_pointcloud(lob_now)
        feats = pointcloud_topology_features(pc, gap_threshold=gap_threshold)

        score = topology_stress_score(
            n_components=feats["n_components"],
            max_gap=feats["max_gap"],
            mean_gap=feats["mean_gap"],
            shape_dist=shape_dist,
        )
        stress_scores.append(score)

    threshold, shift_indices = detect_regime_shifts(
        stress_scores,
        k=regime_k,
    )

    return {
        "series": series,
        "stress_scores": stress_scores,
        "threshold": threshold,
        "shift_indices": shift_indices,
    }
import matplotlib.pyplot as plt

def plot_mte_results(result):
    """
    Plot stress scores and detected regime shifts from MTE engine result.
    """
    stress = result["stress_scores"]
    threshold = result["threshold"]
    shifts = result["shift_indices"]

    plt.figure(figsize=(10, 4))

    # main stress curve
    plt.plot(stress, marker="o", label="Stress score")

    # threshold line
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")

    # highlight regime shifts
    xs = shifts
    ys = [stress[i] for i in shifts]
    plt.scatter(xs, ys, s=80, marker="x", color="black", label="Regime shifts")

    plt.title("Market-Topology Stress Curve with Regime-Shift Flags")
    plt.xlabel("Time step")
    plt.ylabel("Stress score")
    plt.legend()
    plt.tight_layout()
    plt.show()
import os
import pandas as pd

def save_mte_results(result, output_dir: str = "data", filename: str = "mte_results.csv"):
    """
    Save MTE stress scores and regime-shift flags to a CSV file.

    - output_dir: folder to save in (default: 'data')
    - filename: name of the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)

    stress = result["stress_scores"]
    threshold = result["threshold"]
    shifts = set(result["shift_indices"])

    # build a small table
    rows = []
    for i, s in enumerate(stress):
        rows.append({
            "time_index": i,
            "stress_score": s,
            "is_regime_shift": int(i in shifts),
            "threshold": threshold,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved MTE results to {path}")
