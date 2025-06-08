````markdown
# 📦 Federated Averaging (FedAvg) Implementation

## 📌 Strategy Overview

**FedAvg** is the baseline Federated Learning algorithm that:
1. Trains models locally on each client.
2. Sends updated weights to the central server.
3. Averages weights from all participating clients.
4. Repeats for multiple rounds.

### 🧮 Key Equation:

$$`
w_{t+1} = \frac{1}{K} \sum_{k=1}^K w_t^{(k)}
$$

Where:
- $`K`$: Number of clients
- $w_t^{(k)}$: Model weights from client $k$ at round### 🧮 Key Equation:

$$
w_{t+1} = \frac{1}{K} \sum_{k=1}^K w_t^{(k)}
$$

Where:
- $K$: Number of clients
- $w_t^{(k)}$: Model weights from client $k$ at round $t$

 $t$


---

## 🏗️ Branch Contents

This branch contains the full implementation of FedAvg with modular structure and JSON results per heterogeneity level (α):

```
FedAvg/
├── src/
│   ├── client.py       # Local training logic
│   ├── server.py       # Weight averaging
│   └── strategy.py     # FedAvg algorithm implementation
├── configs/
│   └── fedavg.yaml     # Hyperparameters and experiment configs
└── results/
    ├── alpha_10.json   # Results for α = 10
    ├── alpha_1.json    # Results for α = 1
    └── alpha_0.1.json  # Results for α = 0.1
```

---

## 🔍 Core Implementation

### 🔑 Key Files

| File          | Role                 | Code Reference                                       |
| ------------- | -------------------- | ---------------------------------------------------- |
| `strategy.py` | Aggregation logic    | [`src/strategy.py#L45-L72`](src/strategy.py#L45-L72) |
| `client.py`   | Local training (SGD) | [`src/client.py#L88-L112`](src/client.py#L88-L112)   |

### 🧩 Key Snippet (Weight Averaging)

```python
# In strategy.py
def aggregate_fit(self, results):
    weights = [parameters_to_ndarrays(r.parameters) for _, r in results]
    avg_weights = [
        np.mean(layer_weights, axis=0) 
        for layer_weights in zip(*weights)
    ]
    return ndarrays_to_parameters(avg_weights)
```

---

## 📊 Performance Summary by α Values

| α Value | Train Acc | Val Acc | Train Loss | Val Loss |
| ------- | --------- | ------- | ---------- | -------- |
| 10      | 0.92      | 0.86    | 0.21       | 0.45     |
| 1       | 0.88      | 0.81    | 0.34       | 0.62     |
| 0.1     | 0.79      | 0.71    | 0.51       | 0.89     |

---

## 📈 Training Dynamics (Visuals)

### 🔹 α = 10 (Near-IID)

![α=10 Training](results/alpha_10_train.png)

> *Stable convergence with low variance. FedAvg performs optimally under near-IID conditions.*

---

### 🔹 α = 1 (Moderate Heterogeneity)

![α=1 Training](results/alpha_1_train.png)

> *Slightly unstable training. Some client drift appears. Accuracy decreases moderately.*

---

### 🔹 α = 0.1 (High Heterogeneity)

![α=0.1 Training](results/alpha_0.1_train.png)

> *Significant divergence. Training becomes noisy and unstable. FedAvg struggles to converge.*

---

## 💡 Key Observations

1. **Low Heterogeneity (α = 10)**:

   * FedAvg performs reliably.
   * Training is smooth and centralized averaging is effective.

2. **Moderate Heterogeneity (α = 1)**:

   * FedAvg starts to suffer from client drift.
   * Results still usable but suboptimal.

3. **High Heterogeneity (α = 0.1)**:

   * Client updates diverge due to local data imbalance.
   * Training becomes unstable, with much lower validation accuracy.

---

## 🛠️ How to Run

```bash
# Run FedAvg with α = 1
python main.py --alpha 1 --strategy fedavg
```

All hyperparameters are configurable in:

```bash
configs/fedavg.yaml
```

---

## 📝 Conclusion

This branch demonstrates a clean and modular implementation of **Federated Averaging (FedAvg)**. While FedAvg performs well on IID or mildly non-IID data (high α), it **fails to maintain convergence** on highly heterogeneous data (low α), confirming its **sensitivity to client drift**. This motivates the need for more robust strategies like **FedProx** and **SCAFFOLD**, which are explored in other branches of this project.

> For a comparison of FedAvg with FedProx and SCAFFOLD, refer to the [`main` branch](https://github.com/your-repo/tree/main).

