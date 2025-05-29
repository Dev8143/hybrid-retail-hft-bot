#!/usr/bin/env python3
"""Test Feature Engineering (No PyTorch)"""

print('📊 Testing Feature Engineering...')
print('='*50)

try:
    import pandas as pd
    import numpy as np
    import time
    
    class SimpleFeatureEngineer:
        """Simplified feature engineer without PyTorch dependencies"""
        
        def create_features(self, data):
            """Create trading features from OHLCV data"""
            df = data.copy()
            
            # Basic price features
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['spread'] = df['ask'] - df['bid']
            df['spread_pct'] = df['spread'] / df['mid_price'] * 100
            
            # Returns
            df['return_1'] = df['mid_price'].pct_change(1)
            df['return_5'] = df['mid_price'].pct_change(5)
            df['return_10'] = df['mid_price'].pct_change(10)
            
            # Moving averages
            df['ma_5'] = df['mid_price'].rolling(5).mean()
            df['ma_10'] = df['mid_price'].rolling(10).mean()
            df['ma_20'] = df['mid_price'].rolling(20).mean()
            
            # Volatility
            df['volatility_5'] = df['return_1'].rolling(5).std()
            df['volatility_10'] = df['return_1'].rolling(10).std()
            df['volatility_20'] = df['return_1'].rolling(20).std()
            
            # Volume features
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
            df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
            
            # Technical indicators
            # RSI
            delta = df['mid_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['mid_price'].rolling(bb_period).mean()
            bb_std_val = df['mid_price'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_val * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_val * bb_std)
            df['bb_position'] = (df['mid_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum
            df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
            df['momentum_10'] = df['mid_price'] / df['mid_price'].shift(10) - 1
            
            # Spread features
            df['spread_ma_5'] = df['spread'].rolling(5).mean()
            df['spread_ratio'] = df['spread'] / df['spread_ma_5']
            
            # Time-based features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            # Remove rows with NaN values
            df = df.dropna()
            
            return df
    
    print('✓ SimpleFeatureEngineer created')
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    
    price = 1.1000
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        price += np.random.normal(0, 0.0001) - 0.00001 * (price - 1.1000)  # Mean reversion
        prices.append(price)
        volumes.append(np.random.exponential(1000))
    
    data = pd.DataFrame({
        'bid': np.array(prices) - np.random.uniform(0.00001, 0.0001, len(prices)),
        'ask': np.array(prices) + np.random.uniform(0.00001, 0.0001, len(prices)),
        'volume': volumes
    }, index=dates)
    
    print(f'✓ Created sample data: {len(data)} rows')
    
    # Feature engineering
    feature_engineer = SimpleFeatureEngineer()
    start_time = time.time()
    data_with_features = feature_engineer.create_features(data)
    feature_time = time.time() - start_time
    
    print(f'✓ Feature engineering completed in {feature_time:.3f}s')
    print(f'✓ Generated {len(data_with_features.columns)} features')
    print(f'✓ Final dataset: {len(data_with_features)} rows')
    
    # Show some features
    feature_names = list(data_with_features.columns)[:10]
    print(f'✓ Sample features: {", ".join(feature_names)}')
    
    # Test feature quality
    print('\n📈 Testing Feature Quality...')
    
    # Check for NaN values
    nan_count = data_with_features.isnull().sum().sum()
    print(f'✓ NaN values: {nan_count}')
    
    # Check feature statistics
    numeric_features = data_with_features.select_dtypes(include=[np.number])
    print(f'✓ Numeric features: {len(numeric_features.columns)}')
    
    # Show feature ranges
    print('\n📊 Feature Statistics:')
    key_features = ['mid_price', 'spread', 'return_1', 'volatility_10', 'rsi_14', 'volume_ratio_10']
    for feature in key_features:
        if feature in data_with_features.columns:
            values = data_with_features[feature]
            print(f'  {feature}: [{values.min():.6f}, {values.max():.6f}], mean={values.mean():.6f}')
    
    # Test feature computation speed
    print('\n⚡ Testing Feature Computation Speed...')
    
    # Single tick processing speed
    single_tick_data = data.iloc[-50:].copy()  # Last 50 rows
    
    latencies = []
    for i in range(100):
        start_time = time.perf_counter()
        features = feature_engineer.create_features(single_tick_data)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f'✓ Feature computation: {avg_latency:.3f}ms avg, {p95_latency:.3f}ms P95')
    print(f'✓ Processing rate: {1000/avg_latency:.0f} feature sets/second')
    
    # Test incremental feature updates
    print('\n🔄 Testing Incremental Updates...')
    
    # Simulate adding new tick data
    new_tick = pd.DataFrame({
        'bid': [1.1005],
        'ask': [1.1007],
        'volume': [1200]
    }, index=[dates[-1] + pd.Timedelta(minutes=1)])
    
    start_time = time.perf_counter()
    updated_data = pd.concat([data.iloc[-20:], new_tick])  # Keep last 20 + new tick
    updated_features = feature_engineer.create_features(updated_data)
    end_time = time.perf_counter()
    
    incremental_latency = (end_time - start_time) * 1000
    print(f'✓ Incremental update: {incremental_latency:.3f}ms')
    
    # Test signal generation simulation
    print('\n🎯 Testing Signal Generation Logic...')
    
    # Simple signal generation based on features
    signals = []
    confidences = []
    
    for idx, row in data_with_features.iterrows():
        signal = 0
        confidence = 0.5
        
        # RSI-based signals
        if row['rsi_14'] > 70:  # Overbought
            signal = -1
            confidence = min(0.9, 0.5 + (row['rsi_14'] - 70) / 30 * 0.4)
        elif row['rsi_14'] < 30:  # Oversold
            signal = 1
            confidence = min(0.9, 0.5 + (30 - row['rsi_14']) / 30 * 0.4)
        
        # Bollinger Band signals
        if row['bb_position'] > 0.95:  # Near upper band
            signal = -1
            confidence = max(confidence, 0.7)
        elif row['bb_position'] < 0.05:  # Near lower band
            signal = 1
            confidence = max(confidence, 0.7)
        
        # Volume confirmation
        if abs(signal) > 0 and row['volume_ratio_10'] > 1.5:
            confidence = min(0.95, confidence * 1.2)
        
        signals.append(signal)
        confidences.append(confidence)
    
    signals = np.array(signals)
    confidences = np.array(confidences)
    
    signal_count = np.sum(np.abs(signals) > 0)
    buy_signals = np.sum(signals > 0)
    sell_signals = np.sum(signals < 0)
    avg_confidence = np.mean(confidences[np.abs(signals) > 0])
    
    print(f'✓ Generated {signal_count} signals ({buy_signals} buy, {sell_signals} sell)')
    print(f'✓ Average signal confidence: {avg_confidence:.3f}')
    print(f'✓ Signal rate: {signal_count/len(data_with_features)*100:.1f}% of ticks')
    
    print('\n✅ Feature engineering test completed successfully!')
    
except Exception as e:
    print(f'❌ Feature engineering test failed: {e}')
    import traceback
    traceback.print_exc()