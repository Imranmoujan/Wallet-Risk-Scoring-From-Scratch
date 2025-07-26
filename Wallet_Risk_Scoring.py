import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("wallet_data.csv")

# Drop unnecessary columns (not needed for wallet risk analysis)
df.drop([
    "from_address", "to_address", "miner_address", "log_event_index", "event_signature",
    "event_param_from", "event_param_to", "sender_contract_address", "sender_contract_name",
    "sender_contract_symbol", "successful", "event_param_value"
], axis=1, inplace=True)

# Keep only relevant lending protocol events
required_events = [
    'Deposit', 'RedeemUnderlying', 'Withdraw', 'Borrow',
    'Repay', 'RepayBorrow', 'LiquidateBorrow', 'LiquidationCall'
]
df_filtered = df[df['event_name'].isin(required_events)].copy()

# Convert time column to datetime
df_filtered['block_signed_at'] = pd.to_datetime(df_filtered['block_signed_at'])

# Count total number of transactions per wallet
tx_count = df_filtered.groupby('wallet_address').size().rename("total_tx_count")

# Count each type of event per wallet (e.g., how many Borrows, Deposits etc.)
action_counts = df_filtered.groupby(['wallet_address', 'event_name']).size().unstack(fill_value=0)

# Calculate ETH & USD value stats per wallet
volume_stats = df_filtered.groupby('wallet_address').agg({
    'value_eth': ['sum', 'mean', 'max'],
    'value_quote_usd': ['sum', 'mean', 'max']
})
volume_stats.columns = ['_'.join(col) for col in volume_stats.columns]  # Flatten column names

# Gas & fee statistics per wallet
fees_gas_stats = df_filtered.groupby('wallet_address').agg({
    'fees_paid_eth': ['sum', 'mean'],
    'gas_price': 'mean',
    'gas_spent': 'mean'
})
fees_gas_stats.columns = ['_'.join(col) for col in fees_gas_stats.columns]

# First & last transaction and account activity duration
time_stats = df_filtered.groupby('wallet_address').agg(
    first_tx=('block_signed_at', 'min'),
    last_tx=('block_signed_at', 'max'),
    active_days=('block_signed_at', lambda x: x.dt.date.nunique())
)
time_stats['account_age_days'] = (time_stats['last_tx'] - time_stats['first_tx']).dt.days

# Average time between transactions
df_filtered.sort_values(['wallet_address', 'block_signed_at'], inplace=True)
df_filtered['time_diff'] = df_filtered.groupby('wallet_address')['block_signed_at'].diff().dt.total_seconds()
avg_time_gap = df_filtered.groupby('wallet_address')['time_diff'].mean().rename("avg_time_gap")

# Presence of decimals info (if decimals metadata present or not)
decimals_presence = df_filtered.groupby('wallet_address')['sender_contract_decimals'].apply(
    lambda x: x.notnull().mean()
).rename("decimals_presence")

# Combine all features into a single DataFrame
features = pd.concat([
    tx_count, action_counts, volume_stats, fees_gas_stats,
    time_stats, avg_time_gap, decimals_presence
], axis=1).reset_index()

# Drop non-numeric columns before feeding to ML
X = features.drop(columns=['wallet_address', 'first_tx', 'last_tx'])

# Fill missing values (if any)
X['avg_time_gap'] = X['avg_time_gap'].fillna(X['avg_time_gap'].mean())

# Apply Isolation Forest to detect anomalies (risky wallets)
model = IsolationForest(contamination=0.1, random_state=42)  # 10% assumed risky
model.fit(X)

# Get anomaly scores (lower score = more normal, we invert it)
features['risk_score'] = -model.score_samples(X)

# Scale risk score between 0 and 1000 for easier interpretation
scaler = MinMaxScaler(feature_range=(0, 1000))
features['risk_score_scaled'] = scaler.fit_transform(features[['risk_score']])

# Sort wallets by risk (higher score = riskier)
features_sorted = features.sort_values(by='risk_score_scaled', ascending=False).reset_index(drop=True)

# Saveing Wallet Risk Scoring in csv file
features_sorted[["wallet_address","risk_score_scaled"]].to_csv("Wallet_Risk_Scoring.csv", index=False)


# Plot risk score distribution
plt.figure(figsize=(8, 4))
sns.histplot(features['risk_score_scaled'], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title("Wallet Risk Score Distribution")
plt.xlabel("Risk Score (0-1000)")
plt.ylabel("Wallet Count")
plt.tight_layout()
plt.show()

