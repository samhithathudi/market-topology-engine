Market Topology Engine (MTE)

A geometric and topology-inspired framework for understanding structural stress inside limit order books.

What This Project Is

The Market Topology Engine is a research project that explores a simple but powerful idea:

What if a limit order book is treated as a shape, and we study how that shape changes over time?

Instead of viewing the order book as a collection of numbers, MTE represents each snapshot as a geometric object. It then measures how the structure bends, gaps, fragments, stretches, or compresses as the market evolves.

This lets us capture behavior that is invisible when looking only at prices or returns. The goal is not to predict price. The goal is to understand the internal structure of liquidity and how it changes, which is a core theme in market microstructure and quantitative trading research.

⸻

How It Works

1. Synthetic Limit Order Book Generator

MTE creates realistic synthetic limit order books with:
	•	Bid and ask depth across multiple levels
	•	A moving mid-price
	•	Dynamic changes in liquidity

This creates a controlled environment where structural behavior can be isolated and studied.

2. Representing the Order Book as a Shape

Each order book snapshot is converted into:
	•	A point cloud
	•	Bid and ask curves
	•	A simple geometric representation of liquidity

Once converted into a geometric form, the book can be compared across time by analyzing how its structure changes.

3. Extracting Topology-Inspired Features

The system computes features that describe the structure of the book. These include:
	•	Number of connected liquidity regions
	•	Size of the largest gap in price levels
	•	Average spacing between levels
	•	Deformation of the shape from one time to the next

These features are inspired by topological ideas such as connectivity and gaps, while remaining computationally simple and easy to apply to noisy market data.

4. Market Topology Stress Score

All features are combined into a single stress score. High values indicate:
	•	Fragmented liquidity
	•	Large holes in the book
	•	Unusual deformation in structure

This score acts as an indicator of internal market stress.

5. Regime-Shift Detection

A simple detection method identifies when the stress score becomes unusually high. These instances may represent structural regime shifts.
