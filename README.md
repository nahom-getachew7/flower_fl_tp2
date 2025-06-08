# 📦 SCAFFOLD Implementation

## 📌 Strategy Overview

**SCAFFOLD** (Stochastic Controlled Averaging for Federated Learning) is an advanced Federated Learning algorithm that addresses client drift in non-IID settings by using control variates. It:

1. Trains models locally with gradient correction using client and server control variates
2. Sends updated weights and control variate updates to the central server
3. Aggregates weights and updates server control variates
4. Repeats for multiple rounds

### 🧮 Key Equations

**Local Update with Control Variates:**

- Gradient correction: `∇L(w) - c_i + c`
- Where:
  - **∇L(w)**: Local gradient
  - **c_i**: Client control variate
  - **c**: Server control variable

**Control Variate Update:**

- **Δc_i = (w_0 - w_T)/(Kη) - c**
- **c_i^{new} = c_i + Δc_i**
- Where:
  - **w_0**: Initial model weights
  - **w_T**: Updated weights after local training
  - **K**: Number of local steps
  - **η**: Learning rate
  - **c**: Server control variate

**Server Aggregation:**

- **w\_{t+1} = (1/N) \* Σ \[k=1 to N\] w_t^{(k)} \* n_k**
- **c\_{t+1} = c_t + (M/N) \* Σ \[k=1 to M\] Δc_i**
- Where:
  - **N**: Total clients
  - **M**: Sampled clients
  - **n_k**: Number of examples for client k
  - **Δc_i**: Client control variate update

---

## 🔍 Core Implementation

### 🔑 Key Files

| File | Role | Code Reference |
| --- | --- | --- |
| `strategy.py` | SCAFFOLD aggregation logic | `src/strategy.py` |
| `client.py` | Local training with control variates | `src/client.py` |
| `model.py` | CNN model and parameter handling | `src/model.py` |
| `data_utils.py` | Non-IID data partitioning | `src/data_utils.py` |

### 🛠️ Key Snippet (Client Training with Control Variates)

```python
# In client.py
for data, target in self.train_loader:
    data, target = data.to(self.device), target.to(self.device)
    optimizer.zero_grad()
    output = self.model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Apply gradient correction
    with torch.no_grad():
        for param, corr in zip(self.model.parameters(), correction_tensors):
            if param.grad is not None:
                param.grad -= corr  # Subtract (c_i - c)
    
    optimizer.step()
```

### 🛠️ Key Snippet (Server Aggregation)

```python
# In strategy.py
def aggregate_fit(self, server_round, results, failures):
    # Aggregate model parameters
    sum_model_params = [np.zeros_like(c) for c in self.server_control]
    sum_control_updates = [np.zeros_like(c) for c in self.server_control]
    
    for client, fit_res in results:
        res_weights = parameters_to_ndarrays(fit_res.parameters)
        model_params = res_weights[:self.num_model_params]
        client_control_update = res_weights[2*self.num_model_params:3*self.num_model_params]
        
        # Accumulate model parameters (example-weighted)
        for idx, w in enumerate(model_params):
            sum_model_params[idx] += w * fit_res.num_examples
            
        # Accumulate control updates
        for idx, cv in enumerate(client_control_update):
            sum_control_updates[idx] += cv
            
    # Compute averages
    new_global_weights = [param_sum / total_examples for param_sum in sum_model_params]
    avg_control_update = [cv_sum / len(results) for cv_sum in sum_control_updates]
    
    # Update server control
    cv_multiplier = len(results) / total_clients if total_clients > 0 else 1.0
    for idx in range(len(self.server_control)):
        self.server_control[idx] += cv_multiplier * avg_control_update[idx]
```

---

## 📊 Performance Summary by α Values

| α | Train Acc | Val Acc | Train Loss | Val Loss |
| --- | --- | --- | --- | --- |
| 10 | 0.90 | 0.85 | 0.33 | 0.45 |
| 1 | 0.94 | 0.83 | 0.20 | 0.54 |
| 0.1 | 0.93 | 0.81 | 0.19 | 0.57 |

*Note: These are the final results got from the JSON files. we can observe more informations form the graph that is indicated in this report.*

---

## 📈 Training Dynamics (Visuals)

*From the following graph notice the stability of the model while training for 50 rounds*

![Training comparison](Compare/alpha/alpha_comparison.png)

![Training comparison](Compare/alpha/train_val_accuracy_comparison.png)



---

## 💡 Key Observations

1. **Low Heterogeneity (α = 10)**:

   - SCAFFOLD performs comparably to FedAvg but with slightly better stability due to control variates
   - Training is smooth, and aggregation is effective

2. **Moderate Heterogeneity (α = 1)**:

   - SCAFFOLD reduces client drift compared to FedAvg
   - Maintains higher validation accuracy and lower variance in training

3. **High Heterogeneity (α = 0.1)**:

   - SCAFFOLD excels where FedAvg fails, mitigating the impact of non-IID data
   - Control variates ensure more consistent updates, leading to stable convergence

---

## 🛠️ How to Run

```bash
# Generate client data with non-IID partitioning
python main.py generate-data --num-clients 10 --alpha 0.1 
```

```bash
# Run a client (repeat for multiple clients, e.g., cid 0 to 9)
python main.py run-client --cid 0 &
```

```bash
# Run the server
python main.py run-server --address 127.0.0.1:8080 --rounds 3 --output scaffold_results.json
```

```bash
# Run a simulation (combines client and server execution)
python main.py simulate --num-clients 10 --rounds 50 --output scaffold_results_alpha_01.json
```

---

## 📝 Conclusion

This branch provides a modular and robust implementation of **SCAFFOLD**, designed to handle non-IID data in federated learning. By incorporating client and server control variates, SCAFFOLD significantly reduces client drift, achieving **better convergence and stability** compared to FedAvg, especially in highly heterogeneous settings (low α). This implementation serves as a strong baseline for advanced federated learning strategies.

> For a comparison of SCAFFOLD with FedAvg and FedProx, refer to the `main` branch.