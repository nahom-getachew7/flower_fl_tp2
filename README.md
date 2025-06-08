# Federated Learning Strategy Comparison (FedAvg vs FedProx vs SCAFFOLD)

This repository compares three Federated Learning strategies — **FedAvg**, **FedProx**, and **SCAFFOLD** — under varying degrees of data heterogeneity (α ∈ {10, 1, 0.1}) using a non-IID Dirichlet data partitioning.

## 🌍 General Concept

### Why Federated Learning Struggles with Non-IID Data

In real-world FL scenarios, each client may hold data from different distributions. This is known as **non-IID** data. The Dirichlet distribution with parameter **α** is used to simulate such heterogeneity:

- **α = 10**: Low heterogeneity (close to IID)
- **α = 1**: Moderate heterogeneity
- **α = 0.1**: High heterogeneity

With non-IID data, **client drift** becomes a major issue, leading to poor convergence and unstable training in basic strategies like FedAvg.

### Optimization Motivations

- **FedAvg** performs poorly under non-IID conditions due to client drift.
- **FedProx** introduces a proximal term to reduce local model divergence.
- **SCAFFOLD** applies control variates to align local updates to the global objective, reducing variance.

---

## 🧪 What's in the `main` Branch?

This branch is focused **only on comparing** the three strategies. It contains:

- **`src/`**  
  Source code for comparing FedAvg, FedProx, and SCAFFOLD at α = 10, 1, 0.1.

- **`results/`**  
  JSON training history files from simulations done in the three corresponding branches:
  - [`FedAvg`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedAvg)
  - [`FedProx`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedProx)
  - [`SCAFFOLD`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/SCAFFOLD)

- **`comparison_figures/`**  
  Training & validation accuracy graphs for each strategy at each α level:
  - [`comparison_figures/alpha_10/`](Comparison_figures/alpha_10/)
  - [`comparison_figures/alpha_1/`](Comparison_figures/alpha_1/)
  - [`comparison_figures/alpha_0.1/`](Comparison_figures/alpha_0.1/)

---

## 🔁 Contents of Other Branches

- **[`FedAvg`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedAvg)**: Baseline FL algorithm using simple model averaging.
- **[`FedProx`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedProx)**: Introduces a proximal term to reduce divergence.
- **[`SCAFFOLD`](https://github.com/nahom-getachew7/flower_fl_tp2/tree/SCAFFOLD)**: Uses control variates to reduce update variance.

---

## 📊 Comparison of Strategies

### Results Summary Table
Based on validation accuracy
| Strategy | α=10 (IID-like) | α=1 (Moderate) | α=0.1 (High) |
|----------|----------------|----------------|--------------|
| FedAvg   | 0.86          | 0.81         | 0.71       |
| FedProx(mu = 0.1)  | 0.84          | 0.81          | 0.80        |
| SCAFFOLD | 0.85          | 0.83         | O.81       |

Note that I used mu = 0.1 for FedProx for the comparison between the three strategies because in Branch FedProx we can notice that this value(0.1) give a better result compared to the other values(0.5 and 1.0).

## 📉 Visual Analysis and Observations

### α = 10 (Low Heterogeneity)

![α=10 Comparison](Comparison_figures/alpha_10/train_val_accuracy_comparison.png)

- FedAvg and SCAFFOLD perform almost similarly due to the near-IID nature of client data.
- SCAFFOLD shows more stable convergence.
- FedProx is relatively unstabel and lower performance.

### α = 1 (Moderate Heterogeneity)

![α=1 Comparison](Comparison_figures/alpha_1/train_val_accuracy_comparison.png)

- FedAvg starts to show degradation due to client drift.
- The same for SCAFFOLD but there is unstability here too.
- FedProx again lower performance and great unstability.

### α = 0.1 (High Heterogeneity)

![α=0.1 Comparison](Comparison_figures/alpha_0.1/train_val_accuracy_comparison.png)

- FedAvg suffers significantly, validating its sensitivity to non-IID data.
- FedProx shows better improvement, but convergence is **slower**—this might explain why it underperforms in some early-stage experiments.
- SCAFFOLD again delivers the best stability and final accuracy due to its variance reduction mechanism.

---

## ⚠️ Limitations & Constraints

- **Training Time & Resources**: Training on all α values, especially under high heterogeneity, required significant time and machine capacity.  
  Due to these constraints:
  - The main results are derived using **Flower simulation**. It mimics real distributed settings while running faster on a single machine.
  - Real-world distributed implementations were also tested and **executed correctly**.  
    These results are available via additional **`.json` logs** included in each strategy’s branch.

- **FedProx Convergence**:  
  FedProx tends to **converge slower**, especially when μ is small.  
  The 50 communication rounds might not be sufficient for it to reach its full potential. More rounds may yield better results.

- **Hyperparameter Sensitivity**:  
  The choice of μ in FedProx is crucial. While μ = 0.1 gave better results relatively to the others, the optimal value might lie between 0.1 and 0.5 depending on the dataset and heterogeneity level. Further fine-tuning is needed.

---

## ✅ Conclusion

This project shows that **data heterogeneity has a major impact** on federated learning performance:

- **FedAvg** is efficient in IID or near-IID scenarios but fails under strong heterogeneity.
- **FedProx** helps in stabilizing training in heterogeneous cases, though it may need more rounds and better-tuned μ values.
- **SCAFFOLD** consistently outperforms others, especially in highly non-IID setups, due to its correction mechanism for client drift.


---

📂 This README is for the **`main` branch**. Visit the strategy-specific branches for implementation details:

- [FedAvg Branch](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedAvg)
- [FedProx Branch](https://github.com/nahom-getachew7/flower_fl_tp2/tree/FedProx)
- [SCAFFOLD Branch](https://github.com/nahom-getachew7/flower_fl_tp2/tree/SCAFFOLD)

