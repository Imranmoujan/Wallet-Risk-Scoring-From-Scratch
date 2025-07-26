# Wallet-Risk-Scoring-From-Scratch

This project builds a machine learning that analyzes historical on-chain transaction data to assign a **risk score (0–1000)** to wallets interacting with DeFi lending protocols like **Aave V2**.

---

## Dataset Overview

The input file is `wallet_data.csv`, containing historical transaction logs enriched with the following information:

- Wallet address
- Event type (`Deposit`, `Borrow`, `Repay`, etc.)
- Gas and fee statistics
- ETH and USD value transferred
- Timestamps, token metadata, and more

---

## Feature Engineering

We create wallet-level features grouped into several meaningful dimensions:

### 1. **Transaction Activity**
- `total_tx_count`: Total transactions per wallet
- Count of each event type: `Borrow`, `Deposit`, `Repay`, `Withdraw`, `LiquidateBorrow`, etc.

### 2. **Transaction Value**
- `value_eth_sum`, `value_eth_mean`, `value_eth_max`
- `value_quote_usd_sum`, `value_quote_usd_mean`, `value_quote_usd_max`

### 3. **Fees & Gas**
- `fees_paid_eth_sum`, `fees_paid_eth_mean`
- `gas_price_mean`, `gas_spent_mean`

### 4. **Time-based Metrics**
- `first_tx`, `last_tx`: First and last transaction timestamp
- `account_age_days`: Days between first and last activity
- `active_days`: Unique number of active days
- `avg_time_gap`: Average time gap (in seconds) between transactions
  
---

## Modeling: Isolation Forest

We use **`IsolationForest`** from `scikit-learn` to detect anomalous wallets without requiring labeled data.

###  Score Calculation:
- Model outputs anomaly scores (lower = more normal)
- Invert and scale scores using `MinMaxScaler` to **0–1000**
- Higher score → riskier wallet

---
