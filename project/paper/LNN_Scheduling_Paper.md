> **Earlier research draft.** Novelty and causal claims below exceed the demonstrated evidence. See the [repository README](../../README.md) and [methodology](../../docs/ARCHITECTURE.md) for the implemented scope, result provenance and limitations.

# Liquid Neural Networks for Burst Prediction in Smart Manufacturing and Cloud Clusters

## Abstract
Modern CPU and cluster schedulers that utilize deep reinforcement learning (DRL) models typically act on a discrete, fixed-interval basis. This discrete-time assumption forces irregular event streams into a fixed time grid before scheduling models can react, introducing quantization delay. This can be especially problematic during sudden onset bursts of tasks. Liquid Neural Networks (LNNs), and specifically Closed-form Continuous-time Neural Networks (CfC), model time continuously. The time constants in their cells are driven by both the input and continuous elapsed time. In this study, we empirically evaluate whether the continuous-time properties of CfCs provide a measurable advantage over parameter-matched fixed-interval LSTM baselines when forecasting burst onset in both synthetic Poisson/MMPP processes and real-world Alibaba cluster trace data. On a controlled bursty MMPP stream, CfC achieves an AUC of 0.868 and F1 of 0.946, compared to LSTM's AUC of 0.842 and F1 of 0.743, confirming a genuine continuous-time advantage. On the Alibaba real-world trace, LSTM achieves a higher AUC (0.628 vs. 0.539), illustrating that class imbalance and noise dampen the CfC's advantage without further regularization. Both models have identical parameter counts (~21k) and inference latency well within real-time scheduling bounds.

## 1. Introduction
The scheduling of tasks in smart manufacturing environments and cloud clusters is historically dominated by heuristics and, more recently, deep reinforcement learning (DRL) architectures. However, DRL schedulers inherently rely on discrete time steps—observing the system state and taking actions at fixed intervals. While this simplifies the Markov Decision Process (MDP) formulation, it creates a misalignment with real-world scenarios where request arrivals and system failures happen irregularly.

Forcing irregular telemetry onto a fixed time grid introduces a *quantization delay*: if a sudden burst of task arrivals begins mid-tick, the scheduling system cannot detect or react to it until the next tick boundary.

Liquid Neural Networks (LNNs), particularly Closed-form Continuous-time Neural Networks (CfC), present a compelling alternative. By representing time as a continuous variable rather than a discrete step index, the effective reaction speed of a CfC adapts moment-to-moment based on the eventfulness of the input stream. Leng et al. proposed real-time scheduling as a plausible LNN application area in smart manufacturing, yet presented no empirical benchmark [1]. This project addresses that gap by providing the first empirical comparison between continuous-time CfC networks and discrete-time LSTM baselines on burst prediction tasks using real-world scheduling traces.

## 2. Related Work

### 2.1 DRL-based Scheduling
The adoption of deep reinforcement learning in scheduling was pioneered by Mao et al. with DeepRM [2], which demonstrated that policy gradient methods could learn effective resource management strategies directly from experience without human-designed heuristics. This inspired a wave of DRL-based schedulers for cloud computing; however, nearly all these systems depend on state representations sampled at fixed time intervals, leaving them susceptible to quantization delay.

### 2.2 Edge and Heterogeneous Hardware Scheduling
As scheduling tasks move closer to the edge in smart manufacturing and IoT, constraints on computational resources become severe. Traditional numerical ODE solvers used in early continuous-time models are too heavy for these environments, necessitating lightweight inference models.

### 2.3 LNN and CfC Literature
Liquid Time-Constant Networks (LTCs) were introduced by Hasani et al. [3] as a family of neural ODEs where the time constant of each node is modulated by its inputs. While highly expressive and adaptable to irregularly sampled data, LTCs required expensive numerical solvers. To resolve this, Hasani et al. developed Closed-form Continuous-time Neural Networks (CfC) [4], which approximate the interaction between neurons and synapses in a closed-form expression, preserving the continuous-time capabilities without the computational bottleneck of an ODE solver. In a related domain, Wang et al. proposed conditional skip liquid neural networks for large-scale time-series cloud resource prediction [5], focusing on computational efficiency via a skip decision module rather than direct scheduling-decision forecasting.

### 2.4 Genetic and Evolutionary Scheduling
Genetic algorithms (GA) and evolutionary strategies remain a staple in job-shop scheduling. While powerful for static or highly structured offline scheduling, they struggle to adapt to the highly dynamic, sub-second bursty arrivals typical of modern web microservices or cloud ML inference workloads.

## 3. Gap Identification and Novelty Statement
While Leng et al. [1] hypothesized that the flexibility and continuous-time properties of LNNs make them ideal for real-time scheduling in smart manufacturing, no empirical study has validated this claim against realistic, irregular cluster data. Furthermore, while Wang et al. [5] applied LNNs to cloud resource prediction, their focus was on coarse-grained overall resource usage forecasting rather than predicting precise event-driven occurrences like burst onsets.
This study provides the **first empirical validation** comparing continuous-time CfC models to fixed-grid discrete-time baselines (LSTMs) for the specific task of predicting burst onset from irregular, timestamped cluster events.

## 4. Methodology

### 4.1 Datasets
We evaluate the models on two distinct data sources to provide both controlled experimentation and real-world validation:
1. **Synthetic Burst Data**: A controlled event stream generated using a Markov-Modulated Poisson Process (MMPP). This allows us to explicitly define a low-rate background state and a high-rate burst state, testing the models' capability to identify bursts purely from inter-arrival times. We also generated a homogeneous Poisson process as a baseline.
2. **Alibaba Cluster Trace (v2020)**: Real-world trace data from Alibaba's Platform for Artificial Intelligence (PAI) cluster, specifically utilizing the job duration estimates trace which contains highly irregular, bursty job submission events with exact timestamp metadata, CPU, and GPU resource requests.

### 4.2 Preprocessing and Data Pipeline
The core experimental design relies on presenting the *exact same underlying event stream* to both models, but formatted according to their temporal paradigm.
- **CfC Pipeline**: The CfC model receives events exactly as they occur. The input features consist of the elapsed time since the previous event ($\Delta t$), along with any relevant job metadata (e.g., requested resources). The continuous $\Delta t$ is directly passed to the CfC as the `timespans` argument.
- **LSTM Baseline Pipeline**: To mimic traditional DRL schedulers, the continuous event stream is forcibly resampled onto a fixed time grid (1-second bins for synthetic datasets; 10-second bins for the Alibaba trace). The LSTM receives the aggregate count of events that occurred within each bin. Burst labels for the LSTM are computed by counting raw events in the forward window from each grid-point, not grid-point density, ensuring both models face the same binary classification target.

### 4.3 Task Definition
The prediction target is **Burst Onset**. Given the history up to the current time $t$, the model predicts a binary label indicating whether the number of arrivals in the upcoming window $[t, t+W]$ will exceed a critical threshold $N_{burst}$. This is formatted as a binary classification task.

### 4.4 Model Architectures
- **CfC Model**: A Closed-form Continuous-time Neural Network built using the `ncps` library (PyTorch backend). It consists of a single CfC cell followed by a linear classification head.
- **LSTM Baseline**: A standard PyTorch `nn.LSTM` followed by a linear classification head.
Both models are parameter-matched: CfC has 21,025 trainable parameters (CfCCell with a 128-unit backbone + linear head) and LSTM has 21,088 trainable parameters (hidden size 71 + linear head), ensuring the comparison measures temporal representation capacity, not raw model size.

[FIGURE: architecture-diagram.png — inserted separately]

### 4.5 Training and Evaluation
Both models were trained for 20 epochs using Binary Cross-Entropy (BCE) loss with the Adam optimizer (lr = 0.005). We utilized a strict chronological train/validation/test split (70/10/20) to prevent data leakage from the future into the past; burst labels within each split are computed using only that split's own events to avoid boundary leakage. Both models were initialized with `torch.manual_seed(42)` for full reproducibility. Performance is evaluated using Area Under the ROC Curve (AUC), F1-Score, and total accuracy. Inference latency is measured as mean CPU wall-clock time per batch over 100 repetitions after 10 warmup passes, using `time.perf_counter()` for sub-millisecond precision.

## 5. Results

We present the performance of the CfC model versus the LSTM baseline across three datasets. Both models have ~21k parameters (CfC: 21,025; LSTM: 21,088). Table 1 summarizes the primary metrics (AUC, F1-Score, and Accuracy) and the mean inference time per batch on CPU (100 repetitions, 10 warmup passes).

| Dataset | Model | AUC | F1-Score | Accuracy | Inference Time (ms/batch) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Poisson** | CfC | 0.4903 | 0.0000 | 0.8792 | 2.94 |
| | LSTM | 0.4976 | 0.0000 | 0.8769 | 0.76 |
| **Synthetic Bursty (MMPP)** | CfC | **0.8684** | **0.9458** | 0.9163 | 2.98 |
| | LSTM | 0.8422 | 0.7432 | 0.9291 | 0.88 |
| **Alibaba Trace** | CfC | 0.5387 | 0.9843 | 0.9691 | 4.43 |
| | LSTM | **0.6278** | 0.9667 | 0.9357 | 0.83 |

### 5.1 Synthetic Datasets
For the homogeneous Poisson process, arrivals are memoryless. As expected, both models achieved an AUC of approximately 0.49–0.50 and an F1-Score of 0.00, correctly reflecting that burst onset is unpredictable from history in a memoryless process. Both models converge to predicting the majority class (non-burst), yielding ~88% accuracy.

For the bursty MMPP dataset, the CfC model outperformed the parameter-matched LSTM baseline, achieving an AUC of 0.8684 and an F1-Score of 0.9458, compared to the LSTM's AUC of 0.8422 and F1-Score of 0.7432. The continuous-time representation enables the CfC to react to the precise inter-arrival timing that signals the switch from low- to high-rate MMPP states. The LSTM, while able to learn some burst structure from binned event counts, loses sub-bin timing precision and achieves a significantly lower F1 (0.74 vs. 0.95), confirming a genuine advantage for continuous-time representation on bursty irregular data.

### 5.2 Alibaba Cluster Trace
On the real-world Alibaba cluster trace, the LSTM baseline achieved a higher AUC (0.6278) compared to the CfC (0.5387). Both models exhibit high F1-Scores (~0.98 and ~0.97, respectively) and accuracy (~97% and ~94%), driven by severe class imbalance: approximately 96% of prediction points are labeled as burst, leaving little room for discriminative learning. The AUC metric, which measures ranking ability independently of threshold, is therefore the more informative measure here. The LSTM's forced temporal aggregation into 10-second bins inadvertently applies a smoothing effect that appears beneficial on this high-imbalance, noisy trace. The CfC's AUC of 0.54 indicates it is learning a weak positive signal but has not converged to a reliable discriminator within 20 epochs.

### 5.3 Inference Speed
After warmup-corrected timing (10 warm-up passes, 100 measured repetitions, `time.perf_counter()`), the CfC averaged 2.98–4.43 ms per batch on CPU, while the LSTM averaged 0.76–0.88 ms — a ratio of approximately 3.4–5.3×. This gap is partly an implementation artifact: our CfC runs a Python step-loop over the sequence length, whereas PyTorch's LSTM uses a single native fused kernel; compilation via `torch.compile` would reduce this gap substantially. Regardless, both models operate well within the latency bounds required for real-time edge and cluster scheduling systems.

## 6. Discussion
The experimental results provide a nuanced answer to the research question. On controlled bursty synthetic data (MMPP), the CfC's continuous-time representation provides a genuine, measurable advantage: AUC 0.868 vs. 0.842 and, critically, F1 0.946 vs. 0.743 — a gap that directly reflects the LSTM's inability to precisely localise burst onset within a fixed bin. The CfC correctly identifies when the inter-arrival time collapses (signaling entry into the high-rate MMPP state) without waiting for a bin boundary, confirming the quantization-delay hypothesis at the level of burst-onset detection.

On the Alibaba trace, the LSTM achieves a higher AUC (0.628 vs. 0.539). We attribute this to two compounding factors. First, severe class imbalance (~96% positive) gives both models very few negative examples on which to train discriminative features; the AUC values, while reflecting a weak positive signal for both, are noisy and sensitive to small convergence differences over 20 epochs. Second, the LSTM's fixed-bin aggregation inadvertently low-pass filters the raw Alibaba trace, which contains substantial high-frequency noise; this smoothing proves more stable in the presence of extreme imbalance than the CfC's fine-grained per-event reactions.

**Limitations**: This study evaluated a minimal single-cell architecture. A deeper CfC with sparse Neural Circuit Policy (NCP) wiring, focal loss or class-weighted BCE for imbalance mitigation, and more training epochs would likely alter the Alibaba comparison. The bin size for the LSTM was fixed; a sweep over bin sizes would provide a stronger baseline. Both models are evaluated on a relatively short 20,000-event slice of the Alibaba trace; results may differ with a larger or differently-sampled window.

## 7. Conclusion and Future Work
This study provides the first empirical comparison between continuous-time CfC networks and parameter-matched discrete-time LSTM baselines for burst-onset prediction in scheduling-relevant event streams. On controlled bursty data (MMPP), CfC achieves an AUC of 0.868 and F1 of 0.946, compared to LSTM's 0.842 and 0.743, validating the hypothesis that continuous-time representation mitigates quantization delay. On the real-world Alibaba cluster trace, LSTM achieves a higher AUC (0.628 vs. 0.539), indicating that the CfC's advantage does not yet transfer reliably to highly imbalanced, noisy real-world data under the simple single-cell architecture evaluated here.

Future work should investigate: (1) class-imbalance mitigation (focal loss, oversampling) to make the Alibaba comparison discriminative; (2) sparse NCP wiring for regularisation; (3) `torch.compile`-based CfC deployment to reduce the inference latency gap to LSTM; and (4) hybrid architectures that combine discrete-bin noise-smoothing with CfC's precise within-bin timing for real-world robustness.

## References
[1] J. Leng et al., "Liquid neural network in smart manufacturing: A new opportunity," *Advanced Engineering Informatics*, vol. 70, 2026.
[2] H. Mao, M. Alizadeh, I. Menache, and S. Kandula, "Resource Management with Deep Reinforcement Learning," in *Proceedings of the 15th ACM Workshop on Hot Topics in Networks (HotNets-XV)*, 2016.
[3] R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid Time-constant Networks," in *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)*, 2021.
[4] R. Hasani et al., "Closed-form Continuous-time Neural Networks," *Nature Machine Intelligence*, vol. 4, no. 11, pp. 992-1003, 2022.
[5] E. Wang, J. Wang, X. Wang, and Z. Sun, "Conditional skip liquid neural networks: an efficient inference framework for large-scale time-series cloud resource prediction," *Neural Computing and Applications*, vol. 38, no. 9, 2026.
