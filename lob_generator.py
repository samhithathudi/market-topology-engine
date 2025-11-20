import numpy as np
import pandas as pd


def generate_lob_snapshot(
    mid_price: float = 100.0,
    tick_size: float = 0.01,
    num_levels: int = 10,
    base_depth: int = 100
) -> pd.DataFrame:
    """
    Create one synthetic limit order book snapshot.

    Returns a DataFrame with columns:
    ['side', 'price', 'size'] where:
      - side is 'bid' or 'ask'
      - price is the level price
      - size is the order size at that level
    """
    # Create price levels below the mid for bids
    bid_prices = mid_price - np.arange(1, num_levels + 1) * tick_size
    # Create price levels above the mid for asks
    ask_prices = mid_price + np.arange(1, num_levels + 1) * tick_size

    # Random depths around base_depth using a Poisson distribution
    bid_sizes = np.random.poisson(base_depth, size=num_levels)
    ask_sizes = np.random.poisson(base_depth, size=num_levels)

    # Build DataFrames for bids and asks
    bids = pd.DataFrame({
        "side": "bid",
        "price": bid_prices,
        "size": bid_sizes,
    })

    asks = pd.DataFrame({
        "side": "ask",
        "price": ask_prices,
        "size": ask_sizes,
    })

    # Combine into one LOB table
    lob = pd.concat([bids, asks], ignore_index=True)

    # Sort by price (and side just to be consistent)
    lob.sort_values(["price", "side"], ascending=[True, False], inplace=True)
    lob.reset_index(drop=True, inplace=True)

    return lob