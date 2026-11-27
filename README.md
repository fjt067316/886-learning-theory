# Initialization Shapes What Linear Attention Learns In-Context

Code for my CS 886 project. One layer linear self-attention (Zhang, Frei and Bartlett 2024)
trained by gradient flow from unbalanced starts, with infinitely many prompts and with B < d^2 prompts.

## Run

```
pip install -r requirements.txt
python run_population.py    # about 15 s, Section 3 checks and Figure 1
python run_fewprompt.py     # about 3 min, Sections 4 and 5, Figures 2 to 4
```

Numbers go to `results/population.json` and `results/fewprompt.json`, figures go to `figures/`.
Seeds are fixed so reruns give the same numbers.

## Files

- `model.py` model, losses, gradients, closed form end points, ODE solver
- `run_population.py` infinitely many prompts
- `run_fewprompt.py` finitely many prompts, test error vs B, speed vs generalization,
  sharpness, trainable cross-terms, finite step sizes

Setting: d = 4, N = 10, input covariance with eigenvalues (0.5, 4/3, 13/6, 3) in a random basis.
