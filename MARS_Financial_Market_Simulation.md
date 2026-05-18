# MARS: A FINANCIAL MARKET SIMULATION ENGINE POWERED BY GENERATIVE FOUNDATION MODEL

**Authors:** Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu & Jiang Bian (Microsoft Research Asia)
*Published as a conference paper at ICLR 2025*

## ABSTRACT
Generative models aim to simulate realistic effects of various actions across different contexts, from text generation to visual effects. Despite significant efforts to build real-world simulators, the application of generative models to virtual worlds, like financial markets, remains under-explored. In financial markets, generative models can simulate complex market effects of participants with various behaviors, enabling interaction under different market conditions, and training strategies without financial risk.

We propose Large Market Model (LMM), an order-level generative foundation model, for financial market simulation, akin to language modeling in the digital world. Our financial Market Simulation engine (MarS), powered by LMM, addresses the domain-specific need for realistic, interactive and controllable order generation. We showcase MarS as a forecast tool, detection system, analysis platform, and agent training environment.

## 1 INTRODUCTION
The primary aim of generative models is to simulate realistic effects of various actions. The financial market exemplifies a virtual world where each action, from trade execution to strategy deployment, can have ripple effects across a complex network. The ability to model and predict these effects in real time is crucial for traders, analysts, and regulators. Current market simulation models largely lack the resolution, interactivity, and realism needed to reflect the full complexity of order-level behaviors.

To address these gaps, it is crucial to integrate vast amounts of structured financial data, such as Limit Order Book (LOB). We propose the Large Market Model (LMM), a generative foundation model specifically designed for order-level financial market simulation. Powered by LMM, we introduce MarS, unlocking new potential in:
1. **Forecast Tool:** Simulating future market trajectories.
2. **Detection System:** Identifying potential risks not apparent from current observations.
3. **Analysis Platform:** Evaluating the market impact of large orders and answering "what if" questions.
4. **Agent Training Environment:** Training reinforcement learning agents without real-world financial risks.

## 2 MARS DESIGN
MarS excels in three key dimensions: high-resolution, controllability, and interactivity.

### 2.1 Large Market Model for Financial Market Simulation
* **Tokenization of Order and Order-Batch:** LMM models the generation of trading orders as a conditional generation process, using sequential modeling techniques to predict market states.
* **Order Sequence Modeling:** Uses a causal transformer to encode each order and its preceding Limit Order Book (LOB) information as a single token.
* **Order-Batch Sequence Modeling:** Converts order batches into an image-like format, employing VQ-VAE to represent and generate aggregated trading behaviors over discrete time intervals.
* **Ensemble Model:** Combines micro-level behaviors with macro-level market trends.
* **Scaling Law:** LMM's performance improves significantly as the size of the data and the model increases, consistent with scaling laws in other foundation models.

### 2.2 MarS Order Generation Combined with Simulated Clearing House
At the core of MarS is the simulated clearing house, which matches generated and interactive orders in real-time, providing extensive information for subsequent order generation. The blending process adheres to two principles:
* **"Shaping the Future Based on Realized Realities:"** The order-batch model generates the next order-batch based on recent orders and clearing house matching results.
* **"Electing the Best from Every Possible Future:"** Multiple predicted order-batches are generated and the best match to the fine-grained control signal is selected.

## 3 EXPERIMENTS
* **Realistic Simulations:** MarS accurately replicates key stylized facts derived from historical market data, such as Aggregational Gaussianity, Absence of Autocorrelations, and Volatility Clustering.
* **Interactive Simulations:** MarS simulates market impacts by interacting with a trading agent executing a TWAP (Time-Weighted Average Price) strategy. The synthetic data adheres to the Square-Root-Law ($\Delta \propto \sigma\sqrt{Q/V}$).
* **Controllable Simulations:** MarS effectively generates controllable market simulations by replicating historical events using control signals like replay curves and natural language prompts.

## 4 APPLICATIONS
### 4.1 Forecasting
MarS executes simulations at each initial time point and aggregates outcomes to predict trends. LMM-based simulations significantly outperform direct forecasting baselines like DeepLOB.

### 4.2 Detection
By monitoring the similarity between simulated and real market patterns (e.g., spread distributions), MarS can detect market abuse. During market manipulation periods, simulation realism drops significantly, indicating potential anomalies.

### 4.3 "What if" Analysis on Market Impact
MarS acts as a platform to discover new laws explaining market impact. Using symbolic regression, three new factors were discovered: resiliency, LOB pressure, and LOB_depth. MarS models long-term impact decay dynamics using an ordinary differential equation (ODE).

### 4.4 Reinforcement Learning Environment
MarS is ideal for training RL agents. An agent tasked with purchasing a large volume within 5 minutes improved its price advantage from -6 BP to 2-6 BP over the training process.

## 5 RELATED WORK
* **Financial Market Simulation:** Prior works utilized agent-based modeling and Generative Adversarial Networks (GANs). LMM extends this by acting as a generative foundation model tailored for financial tasks.
* **Foundation Models:** GPT-3, LLaMA2, and Sora inspired general world models. In finance, models like FinGPT and BloombergGPT primarily handle NLP tasks. MarS uniquely focuses on order-level market dynamics to build a financial world simulator.

## 6 CONCLUSION
We introduce MarS, an order-level, fine-grained realistic financial market simulation engine powered by the LMM. We presented its scalability, realistic characteristics, and applicability to tasks like forecasting, risk detection, market impact analysis, and RL agent training, demonstrating its potential to catalyze a paradigm shift in financial applications.
