"""
ZeroMQ Bridge for MT5-Python Communication
High-performance communication layer for real-time data exchange
"""

import zmq
import json
import time
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
import logging
from datetime import datetime
import queue
import asyncio
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Market tick data structure"""
    symbol: str
    bid: float
    ask: float
    volume: int
    timestamp: int
    spread: float = 0.0
    
    def __post_init__(self):
        self.spread = self.ask - self.bid

@dataclass
class TradingSignal:
    """Trading signal structure"""
    symbol: str
    signal: int  # 1 for buy, -1 for sell, 0 for no action
    confidence: float
    timestamp: int
    features: List[float]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class PositionInfo:
    """Position information structure"""
    ticket: int
    symbol: str
    type: int  # 0 for buy, 1 for sell
    volume: float
    open_price: float
    current_price: float
    profit: float
    timestamp: int

class ZMQBridge:
    """High-performance ZeroMQ bridge for MT5-Python communication"""
    
    def __init__(self, 
                 data_port: int = 5555,
                 signal_port: int = 5556,
                 monitoring_port: int = 5557,
                 bind_address: str = "tcp://*"):
        
        self.data_port = data_port
        self.signal_port = signal_port
        self.monitoring_port = monitoring_port
        self.bind_address = bind_address
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        self.data_socket = None
        self.signal_socket = None
        self.monitoring_socket = None
        
        # Data queues
        self.tick_queue = queue.Queue(maxsize=10000)
        self.signal_queue = queue.Queue(maxsize=1000)
        self.monitoring_queue = queue.Queue(maxsize=1000)
        
        # Callbacks
        self.tick_callbacks: List[Callable[[MarketTick], None]] = []
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []
        
        # Threading
        self.running = False
        self.threads: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance metrics
        self.tick_count = 0
        self.signal_count = 0
        self.last_tick_time = 0
        self.latency_samples = []
        
    def start(self):
        """Start the ZMQ bridge"""
        logger.info("Starting ZMQ Bridge...")
        
        # Initialize sockets
        self._init_sockets()
        
        # Start worker threads
        self.running = True
        self._start_threads()
        
        logger.info(f"ZMQ Bridge started on ports {self.data_port}, {self.signal_port}, {self.monitoring_port}")
    
    def stop(self):
        """Stop the ZMQ bridge"""
        logger.info("Stopping ZMQ Bridge...")
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Close sockets
        self._close_sockets()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ZMQ Bridge stopped")
    
    def _init_sockets(self):
        """Initialize ZMQ sockets"""
        # Data socket (PULL) - receives market data from MT5
        self.data_socket = self.context.socket(zmq.PULL)
        self.data_socket.bind(f"{self.bind_address}:{self.data_port}")
        self.data_socket.setsockopt(zmq.RCVHWM, 10000)  # High water mark
        
        # Signal socket (PUSH) - sends trading signals to MT5
        self.signal_socket = self.context.socket(zmq.PUSH)
        self.signal_socket.bind(f"{self.bind_address}:{self.signal_port}")
        self.signal_socket.setsockopt(zmq.SNDHWM, 1000)
        
        # Monitoring socket (PUB) - publishes monitoring data
        self.monitoring_socket = self.context.socket(zmq.PUB)
        self.monitoring_socket.bind(f"{self.bind_address}:{self.monitoring_port}")
        
    def _close_sockets(self):
        """Close ZMQ sockets"""
        if self.data_socket:
            self.data_socket.close()
        if self.signal_socket:
            self.signal_socket.close()
        if self.monitoring_socket:
            self.monitoring_socket.close()
        self.context.term()
    
    def _start_threads(self):
        """Start worker threads"""
        # Data receiver thread
        data_thread = threading.Thread(target=self._data_receiver_worker, daemon=True)
        data_thread.start()
        self.threads.append(data_thread)
        
        # Data processor thread
        processor_thread = threading.Thread(target=self._data_processor_worker, daemon=True)
        processor_thread.start()
        self.threads.append(processor_thread)
        
        # Signal sender thread
        signal_thread = threading.Thread(target=self._signal_sender_worker, daemon=True)
        signal_thread.start()
        self.threads.append(signal_thread)
        
        # Monitoring thread
        monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        monitoring_thread.start()
        self.threads.append(monitoring_thread)
    
    def _data_receiver_worker(self):
        """Worker thread for receiving market data"""
        while self.running:
            try:
                # Non-blocking receive with timeout
                if self.data_socket.poll(timeout=100):  # 100ms timeout
                    message = self.data_socket.recv_string(zmq.NOBLOCK)
                    
                    # Parse message
                    data = json.loads(message)
                    tick = MarketTick(**data)
                    
                    # Add to queue
                    if not self.tick_queue.full():
                        self.tick_queue.put(tick)
                        self.tick_count += 1
                    else:
                        logger.warning("Tick queue full, dropping tick")
                        
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Error in data receiver: {e}")
                time.sleep(0.001)
    
    def _data_processor_worker(self):
        """Worker thread for processing market data"""
        while self.running:
            try:
                # Get tick from queue
                tick = self.tick_queue.get(timeout=0.1)
                
                # Calculate latency
                current_time = int(time.time() * 1000)
                latency = current_time - tick.timestamp
                self.latency_samples.append(latency)
                
                # Keep only last 1000 samples
                if len(self.latency_samples) > 1000:
                    self.latency_samples = self.latency_samples[-1000:]
                
                # Process tick with callbacks
                for callback in self.tick_callbacks:
                    try:
                        self.executor.submit(callback, tick)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {e}")
                
                self.last_tick_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in data processor: {e}")
    
    def _signal_sender_worker(self):
        """Worker thread for sending trading signals"""
        while self.running:
            try:
                # Get signal from queue
                signal = self.signal_queue.get(timeout=0.1)
                
                # Convert to JSON
                signal_data = asdict(signal)
                message = json.dumps(signal_data)
                
                # Send signal
                self.signal_socket.send_string(message, zmq.NOBLOCK)
                self.signal_count += 1
                
            except queue.Empty:
                continue
            except zmq.Again:
                logger.warning("Signal socket busy, dropping signal")
            except Exception as e:
                logger.error(f"Error in signal sender: {e}")
    
    def _monitoring_worker(self):
        """Worker thread for publishing monitoring data"""
        while self.running:
            try:
                # Create monitoring data
                monitoring_data = {
                    "timestamp": int(time.time() * 1000),
                    "tick_count": self.tick_count,
                    "signal_count": self.signal_count,
                    "tick_queue_size": self.tick_queue.qsize(),
                    "signal_queue_size": self.signal_queue.qsize(),
                    "avg_latency": np.mean(self.latency_samples) if self.latency_samples else 0,
                    "max_latency": np.max(self.latency_samples) if self.latency_samples else 0,
                    "last_tick_time": self.last_tick_time
                }
                
                # Add custom monitoring data from queue
                while not self.monitoring_queue.empty():
                    try:
                        custom_data = self.monitoring_queue.get_nowait()
                        monitoring_data.update(custom_data)
                    except queue.Empty:
                        break
                
                # Publish monitoring data
                message = json.dumps(monitoring_data)
                self.monitoring_socket.send_string(f"monitoring {message}", zmq.NOBLOCK)
                
                time.sleep(1.0)  # Publish every second
                
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")
    
    def add_tick_callback(self, callback: Callable[[MarketTick], None]):
        """Add callback for tick processing"""
        self.tick_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Add callback for signal processing"""
        self.signal_callbacks.append(callback)
    
    def send_signal(self, signal: TradingSignal):
        """Send trading signal to MT5"""
        if not self.signal_queue.full():
            self.signal_queue.put(signal)
        else:
            logger.warning("Signal queue full, dropping signal")
    
    def add_monitoring_data(self, data: Dict[str, Any]):
        """Add custom monitoring data"""
        if not self.monitoring_queue.full():
            self.monitoring_queue.put(data)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            "tick_count": self.tick_count,
            "signal_count": self.signal_count,
            "avg_latency_ms": np.mean(self.latency_samples) if self.latency_samples else 0,
            "max_latency_ms": np.max(self.latency_samples) if self.latency_samples else 0,
            "min_latency_ms": np.min(self.latency_samples) if self.latency_samples else 0,
            "tick_rate_per_sec": self.tick_count / max(1, (time.time() - self.last_tick_time / 1000))
        }

class HFTDataProcessor:
    """High-frequency data processor with feature extraction"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.tick_buffer = []
        self.feature_cache = {}
        
    def process_tick(self, tick: MarketTick) -> Optional[Dict[str, float]]:
        """Process incoming tick and extract features"""
        # Add to buffer
        self.tick_buffer.append(tick)
        
        # Maintain window size
        if len(self.tick_buffer) > self.window_size:
            self.tick_buffer.pop(0)
        
        # Need minimum data for features
        if len(self.tick_buffer) < 20:
            return None
        
        # Extract features
        features = self._extract_features()
        
        return features
    
    def _extract_features(self) -> Dict[str, float]:
        """Extract trading features from tick buffer"""
        if len(self.tick_buffer) < 2:
            return {}
        
        # Convert to arrays for efficient computation
        bids = np.array([tick.bid for tick in self.tick_buffer])
        asks = np.array([tick.ask for tick in self.tick_buffer])
        volumes = np.array([tick.volume for tick in self.tick_buffer])
        spreads = np.array([tick.spread for tick in self.tick_buffer])
        
        mid_prices = (bids + asks) / 2
        
        features = {}
        
        # Price features
        features['mid_price'] = mid_prices[-1]
        features['bid'] = bids[-1]
        features['ask'] = asks[-1]
        features['spread'] = spreads[-1]
        features['spread_pct'] = spreads[-1] / mid_prices[-1]
        
        # Returns and momentum
        if len(mid_prices) > 1:
            returns = np.diff(mid_prices) / mid_prices[:-1]
            features['return_1'] = returns[-1] if len(returns) > 0 else 0
            
            for period in [5, 10, 20]:
                if len(returns) >= period:
                    features[f'return_{period}'] = np.sum(returns[-period:])
                    features[f'momentum_{period}'] = mid_prices[-1] / mid_prices[-period-1] - 1
        
        # Volatility features
        if len(mid_prices) > 10:
            returns = np.diff(mid_prices) / mid_prices[:-1]
            for period in [10, 20]:
                if len(returns) >= period:
                    features[f'volatility_{period}'] = np.std(returns[-period:])
        
        # Volume features
        features['volume'] = volumes[-1]
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                features[f'volume_ma_{period}'] = np.mean(volumes[-period:])
                features[f'volume_ratio_{period}'] = volumes[-1] / features[f'volume_ma_{period}']
        
        # Spread features
        features['spread_ma_10'] = np.mean(spreads[-10:]) if len(spreads) >= 10 else spreads[-1]
        features['spread_ratio'] = spreads[-1] / features['spread_ma_10']
        
        # Technical indicators
        if len(mid_prices) >= 14:
            features['rsi_14'] = self._calculate_rsi(mid_prices, 14)
        
        # Microstructure features
        features['bid_ask_imbalance'] = (bids[-1] - asks[-1]) / (bids[-1] + asks[-1])
        
        # Time-based features
        current_time = datetime.fromtimestamp(self.tick_buffer[-1].timestamp / 1000)
        features['hour'] = current_time.hour
        features['minute'] = current_time.minute
        features['second'] = current_time.second
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class HFTSignalGenerator:
    """High-frequency trading signal generator"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_buffer = []
        self.sequence_length = 50
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load ONNX model for inference"""
        try:
            import onnxruntime as ort
            import joblib
            
            # Load ONNX model
            self.model = ort.InferenceSession(model_path)
            
            # Load scaler
            scaler_path = model_path.replace('.onnx', '_scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def generate_signal(self, features: Dict[str, float]) -> Optional[TradingSignal]:
        """Generate trading signal from features"""
        if not features:
            return None
        
        # Add features to buffer
        self.feature_buffer.append(features)
        
        # Maintain sequence length
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)
        
        # Need enough data for prediction
        if len(self.feature_buffer) < self.sequence_length:
            return None
        
        # Generate signal
        if self.model and self.scaler:
            signal, confidence = self._ml_signal()
        else:
            signal, confidence = self._rule_based_signal()
        
        if abs(signal) > 0.3:  # Threshold for signal generation
            return TradingSignal(
                symbol=features.get('symbol', 'EURUSD'),
                signal=1 if signal > 0 else -1,
                confidence=confidence,
                timestamp=int(time.time() * 1000),
                features=list(features.values())
            )
        
        return None
    
    def _ml_signal(self) -> tuple[float, float]:
        """Generate ML-based signal"""
        try:
            # Prepare feature matrix
            feature_names = list(self.feature_buffer[0].keys())
            feature_matrix = []
            
            for features in self.feature_buffer:
                row = [features.get(name, 0.0) for name in feature_names]
                feature_matrix.append(row)
            
            feature_matrix = np.array(feature_matrix).reshape(1, self.sequence_length, -1)
            
            # Scale features
            original_shape = feature_matrix.shape
            feature_matrix = feature_matrix.reshape(-1, original_shape[-1])
            feature_matrix = self.scaler.transform(feature_matrix)
            feature_matrix = feature_matrix.reshape(original_shape)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: feature_matrix.astype(np.float32)})
            
            prediction = outputs[0][0][0]
            confidence = outputs[1][0][0]
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"ML inference error: {e}")
            return self._rule_based_signal()
    
    def _rule_based_signal(self) -> tuple[float, float]:
        """Generate rule-based signal as fallback"""
        if len(self.feature_buffer) < 2:
            return 0.0, 0.0
        
        current = self.feature_buffer[-1]
        previous = self.feature_buffer[-2]
        
        signal = 0.0
        confidence = 0.5
        
        # Simple mean reversion strategy
        if 'return_1' in current:
            return_1 = current['return_1']
            if abs(return_1) > 0.001:  # 0.1% threshold
                signal = -return_1 * 10  # Mean reversion
                confidence = min(0.8, abs(return_1) * 1000)
        
        # RSI signal
        if 'rsi_14' in current:
            rsi = current['rsi_14']
            if rsi > 70:
                signal -= 0.3
            elif rsi < 30:
                signal += 0.3
        
        # Volume signal
        if 'volume_ratio_10' in current:
            vol_ratio = current['volume_ratio_10']
            if vol_ratio > 2.0:  # High volume
                confidence *= 1.2
        
        return np.clip(signal, -1.0, 1.0), np.clip(confidence, 0.0, 1.0)

def main():
    """Example usage of ZMQ Bridge"""
    
    # Initialize components
    bridge = ZMQBridge()
    processor = HFTDataProcessor()
    signal_generator = HFTSignalGenerator()
    
    # Define callbacks
    def on_tick(tick: MarketTick):
        """Process incoming tick"""
        features = processor.process_tick(tick)
        if features:
            signal = signal_generator.generate_signal(features)
            if signal:
                bridge.send_signal(signal)
                logger.info(f"Generated signal: {signal.signal} with confidence {signal.confidence:.2f}")
    
    def on_signal(signal: TradingSignal):
        """Process outgoing signal"""
        logger.info(f"Sending signal: {signal}")
    
    # Add callbacks
    bridge.add_tick_callback(on_tick)
    bridge.add_signal_callback(on_signal)
    
    # Start bridge
    bridge.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            stats = bridge.get_performance_stats()
            logger.info(f"Performance: {stats}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        bridge.stop()

if __name__ == "__main__":
    main()