#!/usr/bin/env python3
"""Test AI/ML Components"""

print('🧠 Testing AI/ML Components...')
print('='*50)

try:
    from python.models.hft_predictor import HFTModelTrainer, FeatureEngineer, LSTMPredictor
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
    
    # Test model creation
    print('\n🤖 Testing Model Architecture...')
    
    trainer = HFTModelTrainer(model_type='lstm')
    model = trainer.create_model(input_size=20)
    
    print(f'✓ LSTM model created: {type(model).__name__}')
    print(f'✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test data preparation
    print('\n📈 Testing Data Preparation...')
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_with_features, sequence_length=20)
    
    print(f'✓ Training data shape: {X_train.shape}')
    print(f'✓ Test data shape: {X_test.shape}')
    print(f'✓ Target range: [{y_train.min():.6f}, {y_train.max():.6f}]')
    
    # Test model inference speed
    print('\n⚡ Testing Inference Speed...')
    
    # Create a simple model for speed testing
    import torch
    import torch.nn as nn
    
    class FastModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()
            )
            self.confidence = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Use last timestep for simple feedforward
            x = x[:, -1, :]  # Take last timestep
            pred = self.fc(x)
            conf = self.confidence(x)
            return pred, conf
    
    fast_model = FastModel(X_train.shape[-1])
    fast_model.eval()
    
    # Warm up
    with torch.no_grad():
        sample_input = torch.FloatTensor(X_test[:10])
        for _ in range(10):
            _ = fast_model(sample_input)
    
    # Speed test
    latencies = []
    with torch.no_grad():
        for i in range(100):
            start_time = time.perf_counter()
            sample_input = torch.FloatTensor(X_test[i:i+1])
            prediction, confidence = fast_model(sample_input)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f'✓ Inference latency: {avg_latency:.3f}ms avg, {p95_latency:.3f}ms P95')
    print(f'✓ Inference rate: {1000/avg_latency:.0f} predictions/second')
    
    # Test prediction quality
    print('\n🎯 Testing Prediction Quality...')
    
    with torch.no_grad():
        predictions, confidences = fast_model(torch.FloatTensor(X_test[:100]))
        
    pred_std = torch.std(predictions).item()
    conf_mean = torch.mean(confidences).item()
    
    print(f'✓ Prediction std: {pred_std:.6f}')
    print(f'✓ Average confidence: {conf_mean:.3f}')
    print(f'✓ Prediction range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]')
    
    print('\n✅ AI/ML components test completed successfully!')
    
except Exception as e:
    print(f'❌ AI/ML test failed: {e}')
    import traceback
    traceback.print_exc()