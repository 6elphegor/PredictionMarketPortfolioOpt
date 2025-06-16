# PredictionMarketPortfolioOpt

This project provides an interactive, browser-based tool to calculate the optimal portfolio allocation for a set of prediction markets. It implements a model that aims to maximize the expected logarithmic growth of wealth, a strategy closely related to the Kelly Criterion.

The entire application runs client-side using HTML, CSS, and JavaScript, with [TensorFlow.js](https://www.tensorflow.org/js) powering the optimization algorithm.

## Overview

Given a set of prediction markets, each with a market price for a "YES" outcome, the model takes your personal "true" probabilities for those outcomes as input. It then finds the portfolio—how much capital to allocate to YES and NO positions across all markets—that provides the highest expected growth.

This is useful for traders who believe they have an "edge" (i.e., their estimated probabilities are more accurate than the market's) and want to size their positions systematically.

## Model Assumptions

The model's recommendations are based on a specific mathematical framework and operate under the following key assumptions. Understanding these is crucial for interpreting the results:

1.  **Statistical Independence:** All market events are treated as statistically independent. The outcome of one event does not influence the outcome of another.
2.  **Simultaneous Resolution:** All market events are assumed to resolve at the same time. The model does not account for sequential resolution or re-investing profits from one market before another closes.
3.  **Arbitrage Opportunity:** The model assumes that the market prices may differ from the true underlying probabilities, and that this difference (the "edge") can be exploited for profit.
4.  **Log-Utility:** The primary goal is to maximize `E[log(Final Wealth)]`, not `E[Final Wealth]`. This inherently leads to a more risk-averse strategy that avoids ruin, but may underperform strategies that take on more risk in single instances.
5.  **Frictionless Market:** The model does not account for trading fees, market liquidity (slippage), or bid-ask spreads. The `P_adj` variable is a numerical stability adjustment, not a model for market friction.

## How to Use

No server or complex setup is required.

1.  **Download:** Download the `index.html`, `style.css`, and `script.js` files and place them in the same folder.
2.  **Open:** Open the `index.html` file in any modern web browser (like Firefox, Chrome, or Edge). An internet connection is required to load the TensorFlow.js library.

### Inputs

-   **Market Prices (P):** Enter the comma-separated market prices (from 0 to 1) for the "YES" outcome in each market.
-   **True Probabilities (Y_prob):** Enter your best estimate of the true probabilities for the "YES" outcome in each corresponding market.
-   **Initial Wealth (W):** The total amount of capital you have available to invest.
-   **Optimization Parameters:** These control the training process. The defaults are generally effective, but can be tuned.
    -   *Learning Rate:* Controls how large the steps are during optimization. Lower if the model fails to converge.
    -   *Iterations:* The number of training steps. More steps can lead to a more precise result.
    -   *Batch Size:* The number of Monte Carlo simulations per iteration. Higher values give a more accurate gradient at the cost of performance.

### Outputs

After running the optimization, the tool will display:

-   **Summary:** High-level portfolio decisions, including the fraction of wealth allocated to YES vs. NO bets and how much capital remains in cash.
-   **Results Table:** A detailed breakdown for each event:
    -   **Net Shares:** The final number of YES or NO shares to hold after canceling out opposing pairs. For example, buying 10 YES and 5 NO on the same market is equivalent to buying 5 YES and holding $5 in cash.
    -   **Capital Allocated:** The amount of money tied up in the net share position for that event.
    -   **α_yes / α_no:** The internal allocation fractions determined by the model, showing how it distributes capital within the YES and NO pools.

## Technical Implementation

-   **Frontend:** The interface is built with plain HTML and CSS.
-   **Logic:** All calculations are performed in the browser using JavaScript.
-   **Optimization:** [TensorFlow.js](https://www.tensorflow.org/js) is used for its core capability of automatic differentiation. The model's parameters (`θ_yes`, `θ_no`, `φ_yes`, `φ_no`) are trained via gradient ascent to maximize the objective function.
-   **Optimizer:** The model uses the `tf.train.adam()` optimizer, an adaptive learning rate algorithm well-suited for this type of problem.

## License

This project is released under the [MIT License](LICENSE.md).
