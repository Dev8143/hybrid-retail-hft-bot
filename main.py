#!/usr/bin/env python3
"""
Main HFT Bot Application
Entry point for the Hybrid Retail High-Frequency Trading Bot
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import bot components
from python.execution.zmq_bridge import ZMQBridge, HFTDataProcessor, HFTSignalGenerator
from python.risk.risk_manager import RiskManager, RiskLimits
from python.monitoring.dashboard import HFTDashboard, DataCollector
from python.models.hft_predictor import HFTModelTrainer, FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hft_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HFTBotApplication:
    """Main HFT Bot Application"""
    
    def __init__(self, config_path: str = "config/hft_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Core components
        self.zmq_bridge: Optional[ZMQBridge] = None
        self.data_processor: Optional[HFTDataProcessor] = None
        self.signal_generator: Optional[HFTSignalGenerator] = None
        self.risk_manager: Optional[RiskManager] = None
        self.data_collector: Optional[DataCollector] = None
        self.dashboard: Optional[HFTDashboard] = None
        
        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def initialize_components(self):
        """Initialize all bot components"""
        logger.info("Initializing HFT Bot components...")
        
        # Create directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize risk manager
        self._initialize_risk_manager()
        
        # Initialize data components
        self._initialize_data_components()
        
        # Initialize communication bridge
        self._initialize_zmq_bridge()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        logger.info("All components initialized successfully")
    
    def _initialize_risk_manager(self):
        """Initialize risk management system"""
        logger.info("Initializing risk manager...")
        
        # Create risk limits from config
        limits = RiskLimits()
        risk_config = self.config.get('risk', {})
        
        # Position limits
        pos_limits = risk_config.get('position_limits', {})
        limits.max_position_size = pos_limits.get('max_position_size', 1.0)
        limits.max_total_exposure = pos_limits.get('max_total_exposure', 10.0)
        limits.max_correlation = pos_limits.get('max_correlation', 0.8)
        
        # Loss limits
        loss_limits = risk_config.get('loss_limits', {})
        limits.max_daily_loss = loss_limits.get('max_daily_loss', 1000.0)
        limits.max_position_loss = loss_limits.get('max_position_loss', 100.0)
        limits.max_drawdown_pct = loss_limits.get('max_drawdown_pct', 5.0)
        
        # Performance limits
        perf_limits = risk_config.get('performance_limits', {})
        limits.max_latency_ms = perf_limits.get('max_latency_ms', 100.0)
        limits.min_sharpe_ratio = perf_limits.get('min_sharpe_ratio', 0.5)
        limits.max_portfolio_volatility = perf_limits.get('max_portfolio_volatility', 0.02)
        
        # Trading frequency limits
        trading_config = self.config.get('trading', {})
        limits.max_trades_per_minute = trading_config.get('max_trades_per_minute', 100)
        limits.max_trades_per_hour = trading_config.get('max_trades_per_hour', 5000)
        limits.max_trades_per_day = trading_config.get('max_trades_per_day', 50000)
        
        self.risk_manager = RiskManager(limits)
        
        # Add alert callback
        def on_risk_alert(alert):
            logger.warning(f"RISK ALERT: {alert.level.value} - {alert.message}")
            # Could add additional alert handling here (email, telegram, etc.)
        
        self.risk_manager.add_alert_callback(on_risk_alert)
        
    def _initialize_data_components(self):
        """Initialize data processing components"""
        logger.info("Initializing data components...")
        
        # Data processor
        data_config = self.config.get('data', {})
        buffer_size = data_config.get('tick_buffer_size', 10000)
        self.data_processor = HFTDataProcessor(window_size=buffer_size)
        
        # Signal generator
        ai_config = self.config.get('ai', {})
        model_path = ai_config.get('model_path', 'models/hft_model.onnx')
        
        self.signal_generator = HFTSignalGenerator()
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.signal_generator.load_model(model_path)
                logger.info(f"Loaded AI model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load AI model: {e}")
        else:
            logger.info("No AI model found, using rule-based signals")
    
    def _initialize_zmq_bridge(self):
        """Initialize ZeroMQ communication bridge"""
        logger.info("Initializing ZMQ bridge...")
        
        comm_config = self.config.get('communication', {})
        zmq_config = comm_config.get('zmq', {})
        
        self.zmq_bridge = ZMQBridge(
            data_port=zmq_config.get('data_port', 5555),
            signal_port=zmq_config.get('signal_port', 5556),
            monitoring_port=zmq_config.get('monitoring_port', 5557),
            bind_address=zmq_config.get('bind_address', "tcp://*")
        )
        
        # Add callbacks
        self.zmq_bridge.add_tick_callback(self._on_tick)
        self.zmq_bridge.add_signal_callback(self._on_signal)
    
    def _initialize_monitoring(self):
        """Initialize monitoring and dashboard"""
        logger.info("Initializing monitoring...")
        
        # Data collector
        self.data_collector = DataCollector()
        
        # Dashboard
        self.dashboard = HFTDashboard(self.data_collector)
    
    def _on_tick(self, tick):
        """Handle incoming market tick"""
        try:
            # Process tick for features
            features = self.data_processor.process_tick(tick)
            
            if features and self.signal_generator:
                # Generate trading signal
                signal = self.signal_generator.generate_signal(features)
                
                if signal and self.risk_manager:
                    # Check risk limits
                    can_trade, reason = self.risk_manager.can_open_position(
                        signal.symbol, 0.01, features.get('mid_price', 1.0)
                    )
                    
                    if can_trade:
                        # Send signal
                        self.zmq_bridge.send_signal(signal)
                        logger.info(f"Signal sent: {signal.symbol} {signal.signal} "
                                  f"(confidence: {signal.confidence:.2f})")
                        
                        # Update risk manager
                        self.risk_manager.add_trade(
                            signal.symbol, 0.01, features.get('mid_price', 1.0)
                        )
                    else:
                        logger.debug(f"Trade blocked by risk manager: {reason}")
            
            # Update risk manager with price data
            if self.risk_manager:
                self.risk_manager.update_position(
                    tick.symbol, 0.0, (tick.bid + tick.ask) / 2
                )
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _on_signal(self, signal):
        """Handle outgoing trading signal"""
        logger.debug(f"Signal processed: {signal}")
    
    def start(self):
        """Start the HFT bot"""
        logger.info("Starting HFT Bot...")
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Start risk monitoring
            if self.risk_manager:
                self.risk_manager.start_monitoring()
            
            # Start ZMQ bridge
            if self.zmq_bridge:
                self.zmq_bridge.start()
            
            # Start data collection
            if self.data_collector:
                monitoring_config = self.config.get('monitoring', {})
                zmq_config = self.config.get('communication', {}).get('zmq', {})
                connect_addr = zmq_config.get('connect_address', 'tcp://localhost')
                monitoring_port = zmq_config.get('monitoring_port', 5557)
                
                self.data_collector.start_collection(f"{connect_addr}:{monitoring_port}")
            
            self.running = True
            logger.info("HFT Bot started successfully")
            
            # Keep main thread alive
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1.0)
                
                # Periodic health checks
                self._health_check()
                
        except Exception as e:
            logger.error(f"Error starting HFT Bot: {e}")
            self.shutdown()
    
    def start_dashboard(self):
        """Start the monitoring dashboard in a separate process"""
        logger.info("Starting monitoring dashboard...")
        
        try:
            if self.dashboard:
                monitoring_config = self.config.get('monitoring', {})
                dashboard_config = monitoring_config.get('dashboard', {})
                
                host = dashboard_config.get('host', '0.0.0.0')
                port = dashboard_config.get('port', 12000)
                
                # Run dashboard in separate thread
                dashboard_thread = threading.Thread(
                    target=self.dashboard.run,
                    kwargs={'host': host, 'port': port, 'debug': False},
                    daemon=True
                )
                dashboard_thread.start()
                
                logger.info(f"Dashboard started at http://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def _health_check(self):
        """Perform periodic health checks"""
        try:
            # Check ZMQ bridge performance
            if self.zmq_bridge:
                stats = self.zmq_bridge.get_performance_stats()
                
                # Log performance metrics
                if stats['tick_count'] > 0:
                    logger.debug(f"Performance: {stats['tick_count']} ticks, "
                               f"avg latency: {stats['avg_latency_ms']:.2f}ms")
                
                # Check for high latency
                if stats['avg_latency_ms'] > 100:
                    logger.warning(f"High latency detected: {stats['avg_latency_ms']:.2f}ms")
            
            # Check risk metrics
            if self.risk_manager:
                risk_metrics = self.risk_manager.get_risk_metrics()
                
                # Log risk status
                logger.debug(f"Risk: {risk_metrics['position_count']} positions, "
                           f"exposure: ${risk_metrics['current_exposure']:.2f}, "
                           f"daily P&L: ${risk_metrics['daily_pnl']:.2f}")
                
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def shutdown(self):
        """Shutdown the HFT bot"""
        logger.info("Shutting down HFT Bot...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop components in reverse order
        if self.data_collector:
            self.data_collector.stop_collection()
        
        if self.zmq_bridge:
            self.zmq_bridge.stop()
        
        if self.risk_manager:
            self.risk_manager.stop_monitoring()
        
        logger.info("HFT Bot shutdown complete")
    
    def train_model(self):
        """Train or retrain the AI model"""
        logger.info("Starting model training...")
        
        try:
            # Load training data (implement data loading logic)
            # For now, use sample data
            import pandas as pd
            import numpy as np
            
            # Generate sample training data
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=50000, freq='1min')
            
            price = 1.1000
            prices = []
            volumes = []
            
            for i in range(len(dates)):
                price += np.random.normal(0, 0.0001)
                prices.append(price)
                volumes.append(np.random.exponential(1000))
            
            data = pd.DataFrame({
                'timestamp': dates,
                'bid': np.array(prices) - np.random.uniform(0.00001, 0.0001, len(prices)),
                'ask': np.array(prices) + np.random.uniform(0.00001, 0.0001, len(prices)),
                'volume': volumes
            })
            data.set_index('timestamp', inplace=True)
            
            # Feature engineering
            feature_engineer = FeatureEngineer()
            data_with_features = feature_engineer.create_features(data)
            
            # Train model
            ai_config = self.config.get('ai', {})
            model_type = ai_config.get('model_type', 'lstm')
            
            trainer = HFTModelTrainer(model_type=model_type)
            X_train, X_test, y_train, y_test = trainer.prepare_data(data_with_features)
            
            # Training parameters from config
            epochs = ai_config.get('epochs', 50)
            batch_size = ai_config.get('batch_size', 64)
            learning_rate = ai_config.get('learning_rate', 0.001)
            patience = ai_config.get('patience', 10)
            
            history = trainer.train(
                X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, patience=patience
            )
            
            # Export model
            model_path = ai_config.get('model_path', 'models/hft_model.onnx')
            trainer.export_to_onnx(model_path)
            
            # Save complete model
            trainer.save_model(model_path.replace('.onnx', '.pth'))
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HFT Bot Application')
    parser.add_argument('--config', default='config/hft_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--mode', choices=['run', 'train', 'dashboard', 'all'],
                       default='all', help='Operation mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create application
    app = HFTBotApplication(args.config)
    
    try:
        if args.mode == 'train':
            app.train_model()
        elif args.mode == 'dashboard':
            app.initialize_components()
            app.start_dashboard()
            # Keep dashboard running
            while True:
                time.sleep(1)
        elif args.mode == 'run':
            app.start()
        elif args.mode == 'all':
            # Start dashboard in background
            app.start_dashboard()
            time.sleep(2)  # Give dashboard time to start
            
            # Start main bot
            app.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        app.shutdown()

if __name__ == "__main__":
    main()