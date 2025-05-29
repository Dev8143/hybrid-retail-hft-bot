"""
Comprehensive Test Suite for HFT Bot System
Tests for all major components including latency, accuracy, and risk management
"""

import pytest
import numpy as np
import pandas as pd
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.execution.zmq_bridge import ZMQBridge, HFTDataProcessor, HFTSignalGenerator, MarketTick, TradingSignal
from python.risk.risk_manager import RiskManager, RiskLimits, RiskAlert, AlertType, RiskLevel
from python.models.hft_predictor import HFTModelTrainer, FeatureEngineer, LSTMPredictor
from python.monitoring.dashboard import DataCollector

class TestHFTDataProcessor:
    """Test suite for HFT data processing"""
    
    def setup_method(self):
        """Setup test environment"""
        self.processor = HFTDataProcessor(window_size=100)
        
    def test_tick_processing(self):
        """Test basic tick processing"""
        tick = MarketTick(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            volume=1000,
            timestamp=int(time.time() * 1000)
        )
        
        # Process tick
        features = self.processor.process_tick(tick)
        
        # Should return None for first few ticks
        assert features is None or isinstance(features, dict)
        
        # Add more ticks to get features
        for i in range(25):
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + i * 0.0001,
                ask=1.1002 + i * 0.0001,
                volume=1000 + i * 10,
                timestamp=int(time.time() * 1000) + i * 1000
            )
            features = self.processor.process_tick(tick)
        
        # Should have features now
        assert features is not None
        assert isinstance(features, dict)
        assert 'mid_price' in features
        assert 'spread' in features
        assert 'volume' in features
        
    def test_feature_extraction(self):
        """Test feature extraction accuracy"""
        # Create test data
        ticks = []
        base_price = 1.1000
        
        for i in range(50):
            tick = MarketTick(
                symbol="EURUSD",
                bid=base_price + i * 0.0001,
                ask=base_price + i * 0.0001 + 0.0002,
                volume=1000,
                timestamp=int(time.time() * 1000) + i * 1000
            )
            ticks.append(tick)
            self.processor.process_tick(tick)
        
        # Get final features
        features = self.processor.process_tick(ticks[-1])
        
        assert features is not None
        
        # Check specific feature values
        assert features['mid_price'] == pytest.approx(base_price + 49 * 0.0001 + 0.0001, rel=1e-5)
        assert features['spread'] == pytest.approx(0.0002, rel=1e-5)
        assert features['volume'] == 1000
        
    def test_performance_under_load(self):
        """Test processor performance under high load"""
        start_time = time.time()
        
        # Process 10,000 ticks
        for i in range(10000):
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + np.random.normal(0, 0.0001),
                ask=1.1002 + np.random.normal(0, 0.0001),
                volume=1000,
                timestamp=int(time.time() * 1000) + i
            )
            self.processor.process_tick(tick)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10k ticks in less than 1 second
        assert processing_time < 1.0
        
        # Calculate ticks per second
        tps = 10000 / processing_time
        print(f"Processed {tps:.0f} ticks per second")
        
        # Should handle at least 50k ticks per second
        assert tps > 50000

class TestHFTSignalGenerator:
    """Test suite for signal generation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.generator = HFTSignalGenerator()
        
    def test_signal_generation(self):
        """Test basic signal generation"""
        features = {
            'mid_price': 1.1000,
            'spread': 0.0002,
            'volume': 1000,
            'return_1': 0.001,
            'volatility_10': 0.01,
            'rsi_14': 70.0
        }
        
        signal = self.generator.generate_signal(features)
        
        # May or may not generate signal based on thresholds
        if signal:
            assert isinstance(signal, TradingSignal)
            assert signal.signal in [-1, 1]
            assert 0 <= signal.confidence <= 1
            
    def test_signal_consistency(self):
        """Test signal consistency with same inputs"""
        features = {
            'mid_price': 1.1000,
            'spread': 0.0002,
            'volume': 1000,
            'return_1': 0.005,  # Strong signal
            'volatility_10': 0.01,
            'rsi_14': 80.0  # Overbought
        }
        
        # Generate multiple signals with same features
        signals = []
        for _ in range(10):
            signal = self.generator.generate_signal(features)
            if signal:
                signals.append(signal)
        
        if signals:
            # All signals should be the same
            first_signal = signals[0].signal
            for signal in signals[1:]:
                assert signal.signal == first_signal
                
    def test_signal_latency(self):
        """Test signal generation latency"""
        features = {
            'mid_price': 1.1000,
            'spread': 0.0002,
            'volume': 1000,
            'return_1': 0.001,
            'volatility_10': 0.01,
            'rsi_14': 50.0
        }
        
        # Fill buffer first
        for _ in range(50):
            self.generator.generate_signal(features)
        
        # Measure latency
        start_time = time.perf_counter()
        signal = self.generator.generate_signal(features)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Should generate signal in less than 1ms
        assert latency_ms < 1.0
        print(f"Signal generation latency: {latency_ms:.3f}ms")

class TestRiskManager:
    """Test suite for risk management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.limits = RiskLimits()
        self.limits.max_daily_loss = 100.0
        self.limits.max_drawdown_pct = 5.0
        self.limits.max_total_exposure = 1000.0
        
        self.risk_manager = RiskManager(self.limits)
        
    def test_position_limits(self):
        """Test position limit enforcement"""
        # Should allow first position
        can_trade, reason = self.risk_manager.can_open_position("EURUSD", 0.1, 1.1000)
        assert can_trade
        assert reason == "OK"
        
        # Add position
        self.risk_manager.update_position("EURUSD", 0.1, 1.1000)
        
        # Test exposure limit
        can_trade, reason = self.risk_manager.can_open_position("GBPUSD", 100.0, 1.3000)
        assert not can_trade
        assert "exposure" in reason.lower()
        
    def test_loss_limits(self):
        """Test loss limit enforcement"""
        # Simulate daily loss
        self.risk_manager.daily_pnl = -150.0  # Exceeds limit
        
        can_trade, reason = self.risk_manager.can_open_position("EURUSD", 0.1, 1.1000)
        assert not can_trade
        assert "daily loss" in reason.lower()
        
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        # Simulate equity changes
        self.risk_manager.update_equity(10000.0)  # Peak
        self.risk_manager.update_equity(9500.0)   # 5% drawdown
        
        current_dd = self.risk_manager.drawdown_calculator.get_current_drawdown()
        assert current_dd == pytest.approx(5.0, rel=0.01)
        
        # Should block trading at max drawdown
        can_trade, reason = self.risk_manager.can_open_position("EURUSD", 0.1, 1.1000)
        assert not can_trade
        assert "drawdown" in reason.lower()
        
    def test_alert_generation(self):
        """Test risk alert generation"""
        alerts = []
        
        def alert_callback(alert):
            alerts.append(alert)
            
        self.risk_manager.add_alert_callback(alert_callback)
        self.risk_manager.start_monitoring()
        
        # Trigger alert condition
        self.risk_manager.daily_pnl = -200.0  # Exceeds limit
        
        # Wait for monitoring loop
        time.sleep(2.0)
        
        self.risk_manager.stop_monitoring()
        
        # Should have generated alert
        assert len(alerts) > 0
        assert any(alert.alert_type == AlertType.DAILY_LOSS for alert in alerts)
        
    def test_latency_monitoring(self):
        """Test latency monitoring"""
        # Add latency samples
        for _ in range(100):
            self.risk_manager.add_latency_sample(np.random.exponential(20))
        
        stats = self.risk_manager.latency_monitor.get_latency_stats()
        
        assert 'mean' in stats
        assert 'p95' in stats
        assert 'p99' in stats
        assert stats['mean'] > 0

class TestZMQBridge:
    """Test suite for ZMQ communication bridge"""
    
    def setup_method(self):
        """Setup test environment"""
        # Use different ports for testing
        self.bridge = ZMQBridge(
            data_port=15555,
            signal_port=15556,
            monitoring_port=15557
        )
        
    def teardown_method(self):
        """Cleanup test environment"""
        if hasattr(self, 'bridge'):
            self.bridge.stop()
            
    def test_bridge_initialization(self):
        """Test ZMQ bridge initialization"""
        self.bridge.start()
        
        # Should be running
        assert self.bridge.running
        
        # Should have initialized sockets
        assert self.bridge.data_socket is not None
        assert self.bridge.signal_socket is not None
        assert self.bridge.monitoring_socket is not None
        
    def test_callback_registration(self):
        """Test callback registration"""
        tick_received = []
        signal_received = []
        
        def on_tick(tick):
            tick_received.append(tick)
            
        def on_signal(signal):
            signal_received.append(signal)
            
        self.bridge.add_tick_callback(on_tick)
        self.bridge.add_signal_callback(on_signal)
        
        assert len(self.bridge.tick_callbacks) == 1
        assert len(self.bridge.signal_callbacks) == 1
        
    def test_signal_sending(self):
        """Test signal sending"""
        self.bridge.start()
        
        signal = TradingSignal(
            symbol="EURUSD",
            signal=1,
            confidence=0.8,
            timestamp=int(time.time() * 1000),
            features=[1.0, 2.0, 3.0]
        )
        
        # Should not raise exception
        self.bridge.send_signal(signal)
        
        # Check performance stats
        time.sleep(0.1)  # Allow processing
        stats = self.bridge.get_performance_stats()
        assert stats['signal_count'] >= 1

class TestMLModels:
    """Test suite for ML models"""
    
    def setup_method(self):
        """Setup test environment"""
        self.trainer = HFTModelTrainer(model_type='lstm')
        
    def test_model_creation(self):
        """Test model creation"""
        model = self.trainer.create_model(input_size=20)
        
        assert model is not None
        assert isinstance(model, LSTMPredictor)
        
    def test_data_preparation(self):
        """Test data preparation"""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'bid': np.random.normal(1.1000, 0.001, 1000),
            'ask': np.random.normal(1.1002, 0.001, 1000),
            'volume': np.random.exponential(1000, 1000)
        })
        data.set_index('timestamp', inplace=True)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.create_features(data)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(data_with_features)
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(X_train.shape) == 3  # (samples, sequence, features)
        assert X_train.shape[0] == len(y_train)
        
    def test_feature_engineering(self):
        """Test feature engineering"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')
        
        data = pd.DataFrame({
            'bid': np.random.normal(1.1000, 0.001, 100),
            'ask': np.random.normal(1.1002, 0.001, 100),
            'volume': np.random.exponential(1000, 100)
        }, index=dates)
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_features(data)
        
        # Check required features
        assert 'mid_price' in features.columns
        assert 'spread' in features.columns
        assert 'return_1' in features.columns
        assert 'volatility_10' in features.columns
        assert 'price_change' in features.columns  # Target variable
        
        # Check no infinite or NaN values in final data
        assert not features.isin([np.inf, -np.inf]).any().any()
        assert not features.isna().any().any()

class TestPerformance:
    """Performance and latency tests"""
    
    def test_end_to_end_latency(self):
        """Test end-to-end processing latency"""
        processor = HFTDataProcessor(window_size=100)
        generator = HFTSignalGenerator()
        
        # Warm up
        for i in range(60):
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + i * 0.0001,
                ask=1.1002 + i * 0.0001,
                volume=1000,
                timestamp=int(time.time() * 1000) + i * 1000
            )
            features = processor.process_tick(tick)
            if features:
                generator.generate_signal(features)
        
        # Measure latency
        latencies = []
        
        for i in range(100):
            start_time = time.perf_counter()
            
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + np.random.normal(0, 0.0001),
                ask=1.1002 + np.random.normal(0, 0.0001),
                volume=1000,
                timestamp=int(time.time() * 1000)
            )
            
            features = processor.process_tick(tick)
            if features:
                signal = generator.generate_signal(features)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Mean latency: {mean_latency:.3f}ms")
        print(f"P95 latency: {p95_latency:.3f}ms")
        print(f"P99 latency: {p99_latency:.3f}ms")
        
        # Performance requirements
        assert mean_latency < 5.0  # Mean < 5ms
        assert p95_latency < 10.0  # P95 < 10ms
        assert p99_latency < 20.0  # P99 < 20ms
        
    def test_throughput(self):
        """Test system throughput"""
        processor = HFTDataProcessor(window_size=100)
        
        start_time = time.time()
        processed_count = 0
        
        # Process for 1 second
        while time.time() - start_time < 1.0:
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + np.random.normal(0, 0.0001),
                ask=1.1002 + np.random.normal(0, 0.0001),
                volume=1000,
                timestamp=int(time.time() * 1000)
            )
            processor.process_tick(tick)
            processed_count += 1
        
        throughput = processed_count
        print(f"Throughput: {throughput:,} ticks/second")
        
        # Should handle at least 100k ticks per second
        assert throughput > 100000

class TestIntegration:
    """Integration tests for complete system"""
    
    def test_complete_workflow(self):
        """Test complete trading workflow"""
        # Initialize components
        processor = HFTDataProcessor(window_size=50)
        generator = HFTSignalGenerator()
        risk_manager = RiskManager()
        
        signals_generated = []
        
        def signal_callback(signal):
            signals_generated.append(signal)
        
        # Simulate market data and trading
        for i in range(100):
            # Create tick
            tick = MarketTick(
                symbol="EURUSD",
                bid=1.1000 + np.random.normal(0, 0.001),
                ask=1.1002 + np.random.normal(0, 0.001),
                volume=1000,
                timestamp=int(time.time() * 1000) + i * 1000
            )
            
            # Process tick
            features = processor.process_tick(tick)
            
            if features:
                # Generate signal
                signal = generator.generate_signal(features)
                
                if signal:
                    # Check risk limits
                    can_trade, reason = risk_manager.can_open_position(
                        signal.symbol, 0.01, features['mid_price']
                    )
                    
                    if can_trade:
                        signals_generated.append(signal)
                        
                        # Update risk manager
                        risk_manager.add_trade(
                            signal.symbol, 0.01, features['mid_price']
                        )
        
        # Should have generated some signals
        print(f"Generated {len(signals_generated)} trading signals")
        
        # Verify signal quality
        if signals_generated:
            for signal in signals_generated:
                assert signal.symbol == "EURUSD"
                assert signal.signal in [-1, 1]
                assert 0 <= signal.confidence <= 1
                
    def test_error_handling(self):
        """Test system error handling"""
        processor = HFTDataProcessor(window_size=50)
        
        # Test with invalid tick data
        invalid_tick = MarketTick(
            symbol="",  # Invalid symbol
            bid=0.0,    # Invalid price
            ask=-1.0,   # Invalid price
            volume=-100, # Invalid volume
            timestamp=0  # Invalid timestamp
        )
        
        # Should not raise exception
        try:
            features = processor.process_tick(invalid_tick)
            # Should handle gracefully
        except Exception as e:
            pytest.fail(f"System should handle invalid data gracefully: {e}")

def run_performance_benchmark():
    """Run comprehensive performance benchmark"""
    print("\n" + "="*50)
    print("HFT BOT PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Test data processing performance
    print("\n1. Data Processing Performance:")
    test_perf = TestPerformance()
    test_perf.test_throughput()
    
    # Test latency
    print("\n2. End-to-End Latency:")
    test_perf.test_end_to_end_latency()
    
    # Test signal generation
    print("\n3. Signal Generation Performance:")
    test_signal = TestHFTSignalGenerator()
    test_signal.setup_method()
    test_signal.test_signal_latency()
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)

if __name__ == "__main__":
    # Run performance benchmark
    run_performance_benchmark()
    
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])