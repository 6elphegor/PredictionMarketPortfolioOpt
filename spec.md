# Optimal Prediction Market Trading Model

## Overview
This model optimizes portfolio allocation across multiple prediction markets to maximize expected logarithmic wealth growth.

## Assumptions
1. All market events are statistically independent
2. All events resolve simultaneously
3. Market prices may differ from true probabilities, creating arbitrage opportunities

## Variables

### Market Data
- `P`: Vector of market prices for each event, values in (0, 1)
- `P_adj`: Adjusted prices with `P_adj[i] = max(ε, min(1-ε, P[i]))` where `ε = 1e-6`
- `Y_prob`: Vector of true event probabilities (from accurate AI predictions)
- `Y_outcome`: Vector of realized binary outcomes sampled from Bernoulli(Y_prob)

### Portfolio Parameters
- `W`: Initial wealth (constant)
- `f_yes`: Fraction of wealth allocated to yes positions, in [0, 1]
- `f_no`: Fraction of wealth allocated to no positions, in [0, 1]  

### Position Allocations
- `α_yes`: Vector of allocation fractions across yes positions, sums to 1
- `α_no`: Vector of allocation fractions across no positions, sums to 1
- `n_yes`: Vector of yes share counts for each event
- `n_no`: Vector of no share counts for each event

### Optimization Parameters (unconstrained)
- `θ_yes`, `θ_no`: Real-valued parameters for portfolio fractions
- `φ_yes`: Real-valued vector parameterizing yes allocations
- `φ_no`: Real-valued vector parameterizing no allocations

## Model Equations

### Portfolio Allocation
```
(f_yes, f_no) = softmax(θ_yes, θ_no)
```

### Event-wise Allocations
```
α_yes = softmax(φ_yes)
α_no = softmax(φ_no)
```

### Share Counts
```
n_yes[i] = (α_yes[i] × f_yes × W) / P_adj[i]
n_no[i] = (α_no[i] × f_no × W) / (1 - P_adj[i])
```

### Optional Integer Clamping
After optimization, optionally round to integer shares:
```
n_yes_int[i] = round(n_yes[i])
n_no_int[i] = round(n_no[i])
```

### Profit Calculation
```
profit[i] = (Y_outcome[i] - P[i]) × (n_yes[i] - n_no[i])
G = sum(profit)
```

### Performance Metrics
```
wealth_ratio = (W + G) / W
log_return = ln(wealth_ratio)
```

## Objective Function
Maximize expected log return:
```
F = E[log_return] = E[ln((W + G) / W)]
```
where expectation is over `Y_outcome ~ Bernoulli(Y_prob)`

## Optimization Procedure
1. Initialize parameters: `θ_yes`, `θ_no`, `φ_yes`, `φ_no`
2. For each iteration:
   - Sample batch of size B:
     - For each sample, draw `Y_outcome[i] ~ Bernoulli(Y_prob[i])`
     - Compute `log_return` for this realization
   - Compute gradient of mean log return w.r.t. parameters
   - Update parameters via gradient ascent
3. After convergence, optionally apply integer clamping to share counts

## Implementation Notes
- Use adaptive learning rates for stability
- Consider temperature scaling in softmax for sparsity control
- Monitor for numerical overflow in share calculations when P_adj approaches 0 or 1
- Validate that no position exceeds available wealth after integer rounding
- When computing results to show note that yes shares and no shares cancel so such share pairs should be moved into cash