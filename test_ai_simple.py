#!/usr/bin/env python3
"""Test AI/ML Components (Simplified)"""

print('🧠 Testing AI/ML Components (Simplified)...')
print('='*50)

try:
    from python.models.hft_predictor import FeatureEngineer
    import pandas as pd
    import numpy as np
    import time
    
    print('✓ AI/ML modules imported successfully')
    
    # Test feature engineering
    print('\n📊 Testing Feature Engineering...')
    
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
    feature_engineer = FeatureEngineer()
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
    
    # Feature correlation analysis
    if len(numeric_features.columns) > 1:
        corr_matrix = numeric_features.corr()
        high_corr = (corr_matrix.abs() > 0.9) & (corr_matrix != 1.0)
        high_corr_count = high_corr.sum().sum() // 2  # Divide by 2 for symmetric matrix
        print(f'✓ High correlation pairs (>0.9): {high_corr_count}')
    
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
    
    # Test feature stability
    print('\n🎯 Testing Feature Stability...')
    
    # Check if features are stable across similar inputs
    test_data1 = data.iloc[-100:].copy()
    test_data2 = data.iloc[-100:].copy()
    test_data2.iloc[-1, 0] += 0.00001  # Tiny change
    
    features1 = feature_engineer.create_features(test_data1)
    features2 = feature_engineer.create_features(test_data2)
    
    if len(features1) > 0 and len(features2) > 0:
        # Compare last row features
        last_features1 = features1.iloc[-1]
        last_features2 = features2.iloc[-1]
        
        # Calculate relative differences
        numeric_cols = features1.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            rel_diffs = []
            for col in numeric_cols:
                if last_features1[col] != 0:
                    rel_diff = abs((last_features2[col] - last_features1[col]) / last_features1[col])
                    rel_diffs.append(rel_diff)
            
            if rel_diffs:
                max_rel_diff = max(rel_diffs)
                print(f'✓ Feature stability: max relative change {max_rel_diff:.6f}')
    
    print('\n✅ AI/ML components test completed successfully!')
    
except Exception as e:
    print(f'❌ AI/ML test failed: {e}')
    import traceback
    traceback.print_exc()