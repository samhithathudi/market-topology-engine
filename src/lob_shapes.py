import pandas as pd
import matplotlib.pyplot as plt


def plot_lob_snapshot(lob: pd.DataFrame, title: str = "Synthetic LOB snapshot") -> None:
    """
    Plot a simple view of the limit order book snapshot.

    - x-axis: price levels
    - y-axis: size (depth at each level)
    """
    # Separate bids and asks and sort them by price
    bids = lob[lob["side"] == "bid"].sort_values("price")
    asks = lob[lob["side"] == "ask"].sort_values("price")

    plt.figure(figsize=(8, 4))

    # Plot bids and asks as lines with markers
    plt.plot(bids["price"], bids["size"], marker="o", label="bids")
    plt.plot(asks["price"], asks["size"], marker="o", label="asks")

    plt.xlabel("Price")
    plt.ylabel("Size")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np

def snapshot_to_shape(lob):
    """
    Convert one LOB snapshot into 4 arrays:
    - bid_prices
    - bid_sizes
    - ask_prices
    - ask_sizes
    These form the basic geometric 'shape' of the snapshot.
    """
    bids = lob[lob["side"] == "bid"].sort_values("price")
    asks = lob[lob["side"] == "ask"].sort_values("price")

    bid_prices = bids["price"].to_numpy()
    bid_sizes  = bids["size"].to_numpy()

    ask_prices = asks["price"].to_numpy()
    ask_sizes  = asks["size"].to_numpy()

    return bid_prices, bid_sizes, ask_prices, ask_sizes
def lob_shape_distance(lob1, lob2):
    """
    Compute a simple distance between the shapes of two LOB snapshots.

    Steps:
    - Build a common set of price levels from both snapshots.
    - Align bid/ask sizes on that grid (fill missing with 0).
    - Compute an L2 distance between the depth vectors.
    """
    # Build common price grid
    prices1 = lob1["price"].unique()
    prices2 = lob2["price"].unique()
    all_prices = np.union1d(prices1, prices2)

    def depth_vector(lob, side):
        side_df = lob[lob["side"] == side].set_index("price")["size"]
        # align on all_prices, fill missing levels with 0
        aligned = side_df.reindex(all_prices, fill_value=0)
        return aligned.to_numpy()

    bid1 = depth_vector(lob1, "bid")
    ask1 = depth_vector(lob1, "ask")
    bid2 = depth_vector(lob2, "bid")
    ask2 = depth_vector(lob2, "ask")

    # concatenate bid and ask vectors
    vec1 = np.concatenate([bid1, ask1])
    vec2 = np.concatenate([bid2, ask2])

    # Euclidean (L2) distance
    dist = np.linalg.norm(vec1 - vec2)
    return dist
def lob_to_pointcloud(lob):
    """
    Convert a single LOB snapshot into a 2D point cloud.

    Each point is (price, size).
    Bids and asks are merged into one cloud.
    """
    prices = lob["price"].to_numpy()
    sizes = lob["size"].to_numpy()

    # shape (N, 2)
    pointcloud = np.column_stack([prices, sizes])

    return pointcloud

def pointcloud_topology_features(pc, gap_threshold: float = 0.02):
    """
    Very simple 'topology-like' features from a 2D point cloud (price, size).

    Idea:
    - Sort points by price.
    - Look at gaps between neighbouring prices.
    - Large gaps represent 'holes' in liquidity.
    - Continuous runs between large gaps are 'components'.

    Returns a dict with:
      - n_components: number of connected price regions
      - max_gap: largest price gap between neighbouring points
      - mean_gap: average price gap
    """
    # pc is shape (N, 2): [price, size]
    if pc.shape[0] < 2:
        return {
            "n_components": 1,
            "max_gap": 0.0,
            "mean_gap": 0.0,
        }

    # sort by price
    prices = np.sort(pc[:, 0])

    # compute gaps between consecutive prices
    gaps = np.diff(prices)

    max_gap = float(np.max(gaps))
    mean_gap = float(np.mean(gaps))

    # components: count segments separated by gaps bigger than threshold
    n_components = 1
    for g in gaps:
        if g > gap_threshold:
            n_components += 1

    return {
        "n_components": n_components,
        "max_gap": max_gap,
        "mean_gap": mean_gap,
    }
def topology_stress_score(
    n_components: int,
    max_gap: float,
    mean_gap: float,
    shape_dist: float
):
    """
    Combine topology-like features + shape distance into a single market stress score.

    High stress = 
      - more components (fragmented liquidity)
      - larger max gap (big holes)
      - larger shape distance (big deformation)
    """
    # Weighting scheme (you can tune these later)
    w_components = 1.0
    w_maxgap     = 50.0      # price gaps are small numbers, so boost weight
    w_meangap    = 20.0
    w_shape      = 1.0

    score = (
        w_components * n_components +
        w_maxgap * max_gap +
        w_meangap * mean_gap +
        w_shape * shape_dist
    )

    return float(score)
def detect_regime_shifts(stress_scores, k: float = 2.0):
    """
    Detect potential regime shifts in a stress-score time series.

    - stress_scores: list or 1D array of stress values over time
    - k: how many standard deviations above the mean counts as 'high stress'

    Returns:
      - threshold: the numeric cutoff used
      - shift_indices: list of time indices where stress >= threshold
    """
    scores = np.asarray(stress_scores, dtype=float)

    mean = float(np.mean(scores))
    std = float(np.std(scores))

    threshold = mean + k * std

    shift_indices = [i for i, s in enumerate(scores) if s >= threshold]

    return threshold, shift_indices
