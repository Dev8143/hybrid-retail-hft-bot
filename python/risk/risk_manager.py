"""
Advanced Risk Management System for HFT Bot
Comprehensive risk controls and monitoring for high-frequency trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import threading
import time
import json
from enum import Enum
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert type enumeration"""
    POSITION_LIMIT = "position_limit"
    DRAWDOWN = "drawdown"
    DAILY_LOSS = "daily_loss"
    LATENCY = "latency"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SYSTEM_ERROR = "system_error"

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PositionRisk:
    """Position risk metrics"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    var_1d: float  # 1-day Value at Risk
    max_loss: float
    correlation_risk: float
    liquidity_risk: float

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    portfolio_var: float
    max_drawdown: float
    sharpe_ratio: float
    correlation_matrix: np.ndarray
    concentration_risk: float

class RiskLimits:
    """Risk limits configuration"""
    
    def __init__(self):
        # Position limits
        self.max_position_size = 1.0  # Maximum position size per symbol
        self.max_total_exposure = 10.0  # Maximum total exposure
        self.max_positions_per_symbol = 5  # Maximum positions per symbol
        self.max_total_positions = 50  # Maximum total positions
        
        # Loss limits
        self.max_daily_loss = 1000.0  # Maximum daily loss ($)
        self.max_drawdown_pct = 5.0  # Maximum drawdown (%)
        self.max_position_loss = 100.0  # Maximum loss per position ($)
        
        # Performance limits
        self.max_latency_ms = 100.0  # Maximum acceptable latency
        self.min_sharpe_ratio = 0.5  # Minimum Sharpe ratio
        self.max_correlation = 0.8  # Maximum correlation between positions
        
        # Volatility limits
        self.max_portfolio_volatility = 0.02  # Maximum portfolio volatility (2%)
        self.max_position_volatility = 0.05  # Maximum position volatility (5%)
        
        # Frequency limits
        self.max_trades_per_minute = 100  # Maximum trades per minute
        self.max_trades_per_hour = 5000  # Maximum trades per hour
        self.max_trades_per_day = 50000  # Maximum trades per day

class VolatilityEstimator:
    """Real-time volatility estimation"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.return_history = defaultdict(lambda: deque(maxlen=window_size))
        
    def update_price(self, symbol: str, price: float, timestamp: datetime):
        """Update price and calculate returns"""
        prices = self.price_history[symbol]
        returns = self.return_history[symbol]
        
        if len(prices) > 0:
            ret = (price - prices[-1]) / prices[-1]
            returns.append(ret)
        
        prices.append(price)
    
    def get_volatility(self, symbol: str, periods: int = None) -> float:
        """Calculate realized volatility"""
        if periods is None:
            periods = min(len(self.return_history[symbol]), self.window_size)
        
        returns = list(self.return_history[symbol])[-periods:]
        
        if len(returns) < 2:
            return 0.0
        
        return np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized for minute data
    
    def get_garch_volatility(self, symbol: str) -> float:
        """Calculate GARCH-based volatility forecast"""
        returns = list(self.return_history[symbol])
        
        if len(returns) < 10:
            return self.get_volatility(symbol)
        
        # Simple GARCH(1,1) approximation
        returns = np.array(returns)
        
        # Parameters (could be estimated)
        omega = 0.000001
        alpha = 0.1
        beta = 0.85
        
        # Initialize
        sigma2 = np.var(returns)
        
        # Update
        for ret in returns[-10:]:
            sigma2 = omega + alpha * ret**2 + beta * sigma2
        
        return np.sqrt(sigma2) * np.sqrt(252 * 24 * 60)

class CorrelationMonitor:
    """Real-time correlation monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.return_matrix = defaultdict(lambda: deque(maxlen=window_size))
        
    def update_returns(self, symbol: str, return_value: float):
        """Update return for symbol"""
        self.return_matrix[symbol].append(return_value)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix"""
        symbols = list(self.return_matrix.keys())
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        # Create return matrix
        min_length = min(len(self.return_matrix[symbol]) for symbol in symbols)
        
        if min_length < 10:
            return pd.DataFrame()
        
        data = {}
        for symbol in symbols:
            data[symbol] = list(self.return_matrix[symbol])[-min_length:]
        
        df = pd.DataFrame(data)
        return df.corr()
    
    def get_max_correlation(self, exclude_self: bool = True) -> float:
        """Get maximum correlation between any two assets"""
        corr_matrix = self.get_correlation_matrix()
        
        if corr_matrix.empty:
            return 0.0
        
        if exclude_self:
            # Set diagonal to NaN
            np.fill_diagonal(corr_matrix.values, np.nan)
        
        return np.nanmax(corr_matrix.values)

class DrawdownCalculator:
    """Real-time drawdown calculation"""
    
    def __init__(self):
        self.peak_equity = 0.0
        self.equity_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=10000)
        
    def update_equity(self, equity: float, timestamp: datetime):
        """Update equity and calculate drawdown"""
        self.equity_history.append((timestamp, equity))
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.drawdown_history.append((timestamp, drawdown))
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        if not self.drawdown_history:
            return 0.0
        return self.drawdown_history[-1][1] * 100
    
    def get_max_drawdown(self, period_hours: int = 24) -> float:
        """Get maximum drawdown over specified period"""
        if not self.drawdown_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        recent_drawdowns = [dd for ts, dd in self.drawdown_history if ts >= cutoff_time]
        
        return max(recent_drawdowns) * 100 if recent_drawdowns else 0.0

class LatencyMonitor:
    """Real-time latency monitoring"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latency_samples = deque(maxlen=window_size)
        self.latency_by_type = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_latency_sample(self, latency_ms: float, operation_type: str = "general"):
        """Add latency sample"""
        self.latency_samples.append(latency_ms)
        self.latency_by_type[operation_type].append(latency_ms)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latency_samples:
            return {}
        
        samples = list(self.latency_samples)
        
        return {
            "mean": np.mean(samples),
            "median": np.median(samples),
            "p95": np.percentile(samples, 95),
            "p99": np.percentile(samples, 99),
            "max": np.max(samples),
            "std": np.std(samples)
        }
    
    def is_latency_acceptable(self, threshold_ms: float) -> bool:
        """Check if current latency is acceptable"""
        if not self.latency_samples:
            return True
        
        recent_samples = list(self.latency_samples)[-10:]  # Last 10 samples
        return np.mean(recent_samples) <= threshold_ms

class RiskManager:
    """Main risk management system"""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        
        # Risk monitoring components
        self.volatility_estimator = VolatilityEstimator()
        self.correlation_monitor = CorrelationMonitor()
        self.drawdown_calculator = DrawdownCalculator()
        self.latency_monitor = LatencyMonitor()
        
        # Position tracking
        self.positions = {}  # symbol -> position info
        self.position_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=10000)
        
        # Risk metrics
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.trade_count_minute = deque(maxlen=60)
        self.trade_count_hour = deque(maxlen=3600)
        self.trade_count_day = 0
        
        # Alerts
        self.alerts = deque(maxlen=1000)
        self.alert_callbacks = []
        
        # Threading
        self.lock = threading.RLock()
        self.monitoring_thread = None
        self.running = False
        
    def start_monitoring(self):
        """Start risk monitoring thread"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_all_risks()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                time.sleep(1.0)
    
    def _check_all_risks(self):
        """Check all risk conditions"""
        with self.lock:
            # Check position limits
            self._check_position_limits()
            
            # Check loss limits
            self._check_loss_limits()
            
            # Check performance limits
            self._check_performance_limits()
            
            # Check correlation limits
            self._check_correlation_limits()
            
            # Check volatility limits
            self._check_volatility_limits()
            
            # Check frequency limits
            self._check_frequency_limits()
    
    def _check_position_limits(self):
        """Check position-related limits"""
        total_positions = len(self.positions)
        
        if total_positions > self.limits.max_total_positions:
            self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.HIGH,
                f"Total positions ({total_positions}) exceed limit ({self.limits.max_total_positions})"
            )
        
        if self.current_exposure > self.limits.max_total_exposure:
            self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.HIGH,
                f"Total exposure ({self.current_exposure:.2f}) exceeds limit ({self.limits.max_total_exposure})"
            )
    
    def _check_loss_limits(self):
        """Check loss-related limits"""
        current_drawdown = self.drawdown_calculator.get_current_drawdown()
        
        if current_drawdown > self.limits.max_drawdown_pct:
            self._create_alert(
                AlertType.DRAWDOWN,
                RiskLevel.CRITICAL,
                f"Drawdown ({current_drawdown:.2f}%) exceeds limit ({self.limits.max_drawdown_pct}%)"
            )
        
        if self.daily_pnl < -self.limits.max_daily_loss:
            self._create_alert(
                AlertType.DAILY_LOSS,
                RiskLevel.CRITICAL,
                f"Daily loss ({self.daily_pnl:.2f}) exceeds limit ({self.limits.max_daily_loss})"
            )
    
    def _check_performance_limits(self):
        """Check performance-related limits"""
        latency_stats = self.latency_monitor.get_latency_stats()
        
        if latency_stats and latency_stats.get('mean', 0) > self.limits.max_latency_ms:
            self._create_alert(
                AlertType.LATENCY,
                RiskLevel.MEDIUM,
                f"Average latency ({latency_stats['mean']:.2f}ms) exceeds limit ({self.limits.max_latency_ms}ms)"
            )
    
    def _check_correlation_limits(self):
        """Check correlation limits"""
        max_correlation = self.correlation_monitor.get_max_correlation()
        
        if max_correlation > self.limits.max_correlation:
            self._create_alert(
                AlertType.CORRELATION,
                RiskLevel.MEDIUM,
                f"Maximum correlation ({max_correlation:.2f}) exceeds limit ({self.limits.max_correlation})"
            )
    
    def _check_volatility_limits(self):
        """Check volatility limits"""
        # Check individual position volatilities
        for symbol in self.positions:
            vol = self.volatility_estimator.get_volatility(symbol)
            if vol > self.limits.max_position_volatility:
                self._create_alert(
                    AlertType.VOLATILITY,
                    RiskLevel.MEDIUM,
                    f"Volatility for {symbol} ({vol:.4f}) exceeds limit ({self.limits.max_position_volatility})"
                )
    
    def _check_frequency_limits(self):
        """Check trading frequency limits"""
        current_time = time.time()
        
        # Count trades in last minute
        minute_trades = sum(1 for ts in self.trade_count_minute if current_time - ts <= 60)
        if minute_trades > self.limits.max_trades_per_minute:
            self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.HIGH,
                f"Trades per minute ({minute_trades}) exceed limit ({self.limits.max_trades_per_minute})"
            )
        
        # Count trades in last hour
        hour_trades = sum(1 for ts in self.trade_count_hour if current_time - ts <= 3600)
        if hour_trades > self.limits.max_trades_per_hour:
            self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.HIGH,
                f"Trades per hour ({hour_trades}) exceed limit ({self.limits.max_trades_per_hour})"
            )
    
    def _create_alert(self, alert_type: AlertType, level: RiskLevel, message: str, data: Dict = None):
        """Create and process risk alert"""
        alert = RiskAlert(
            alert_type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"RISK ALERT [{level.value.upper()}]: {message}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def update_position(self, symbol: str, size: float, price: float, timestamp: datetime = None):
        """Update position information"""
        with self.lock:
            timestamp = timestamp or datetime.now()
            
            self.positions[symbol] = {
                'size': size,
                'price': price,
                'timestamp': timestamp,
                'market_value': size * price
            }
            
            # Update exposure
            self.current_exposure = sum(abs(pos['market_value']) for pos in self.positions.values())
            
            # Update volatility estimator
            self.volatility_estimator.update_price(symbol, price, timestamp)
    
    def add_trade(self, symbol: str, size: float, price: float, pnl: float = 0.0):
        """Record a new trade"""
        with self.lock:
            current_time = time.time()
            
            # Add to trade counts
            self.trade_count_minute.append(current_time)
            self.trade_count_hour.append(current_time)
            self.trade_count_day += 1
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'size': size,
                'price': price,
                'pnl': pnl,
                'timestamp': datetime.now()
            }
            self.trade_history.append(trade_record)
    
    def update_equity(self, equity: float):
        """Update account equity"""
        with self.lock:
            self.drawdown_calculator.update_equity(equity, datetime.now())
    
    def add_latency_sample(self, latency_ms: float, operation_type: str = "general"):
        """Add latency measurement"""
        self.latency_monitor.add_latency_sample(latency_ms, operation_type)
    
    def can_open_position(self, symbol: str, size: float, price: float) -> Tuple[bool, str]:
        """Check if position can be opened"""
        with self.lock:
            # Check position count
            if len(self.positions) >= self.limits.max_total_positions:
                return False, "Maximum total positions reached"
            
            # Check exposure
            new_exposure = self.current_exposure + abs(size * price)
            if new_exposure > self.limits.max_total_exposure:
                return False, "Maximum total exposure would be exceeded"
            
            # Check individual position size
            if abs(size) > self.limits.max_position_size:
                return False, "Position size exceeds limit"
            
            # Check drawdown
            current_drawdown = self.drawdown_calculator.get_current_drawdown()
            if current_drawdown > self.limits.max_drawdown_pct:
                return False, "Maximum drawdown exceeded"
            
            # Check daily loss
            if self.daily_pnl < -self.limits.max_daily_loss:
                return False, "Daily loss limit exceeded"
            
            # Check trading frequency
            current_time = time.time()
            minute_trades = sum(1 for ts in self.trade_count_minute if current_time - ts <= 60)
            if minute_trades >= self.limits.max_trades_per_minute:
                return False, "Trading frequency limit exceeded"
            
            return True, "OK"
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        with self.lock:
            return {
                'current_exposure': self.current_exposure,
                'position_count': len(self.positions),
                'daily_pnl': self.daily_pnl,
                'current_drawdown': self.drawdown_calculator.get_current_drawdown(),
                'max_drawdown_24h': self.drawdown_calculator.get_max_drawdown(24),
                'latency_stats': self.latency_monitor.get_latency_stats(),
                'max_correlation': self.correlation_monitor.get_max_correlation(),
                'trade_count_day': self.trade_count_day,
                'recent_alerts': len([a for a in self.alerts if 
                                    (datetime.now() - a.timestamp).seconds < 3600])
            }
    
    def get_position_risk(self, symbol: str) -> Optional[PositionRisk]:
        """Get risk metrics for specific position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        volatility = self.volatility_estimator.get_volatility(symbol)
        
        # Calculate VaR (simplified)
        var_1d = abs(position['market_value']) * volatility * 2.33  # 99% confidence
        
        return PositionRisk(
            symbol=symbol,
            position_size=position['size'],
            market_value=position['market_value'],
            unrealized_pnl=0.0,  # Would need current market price
            var_1d=var_1d,
            max_loss=var_1d,
            correlation_risk=0.0,  # Would need correlation calculation
            liquidity_risk=0.0   # Would need liquidity metrics
        )
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of each day)"""
        with self.lock:
            self.daily_pnl = 0.0
            self.trade_count_day = 0
            logger.info("Daily risk counters reset")

def main():
    """Example usage of risk manager"""
    
    # Create risk manager
    risk_manager = RiskManager()
    
    # Add alert callback
    def on_alert(alert: RiskAlert):
        print(f"ALERT: {alert.level.value} - {alert.message}")
    
    risk_manager.add_alert_callback(on_alert)
    
    # Start monitoring
    risk_manager.start_monitoring()
    
    try:
        # Simulate some trading activity
        for i in range(100):
            # Simulate position updates
            risk_manager.update_position(f"SYMBOL_{i%5}", 0.1, 100.0 + i)
            
            # Simulate trades
            risk_manager.add_trade(f"SYMBOL_{i%5}", 0.1, 100.0 + i, np.random.normal(0, 10))
            
            # Simulate equity updates
            risk_manager.update_equity(10000 + np.random.normal(0, 100))
            
            # Simulate latency
            risk_manager.add_latency_sample(np.random.exponential(20))
            
            time.sleep(0.1)
        
        # Print risk metrics
        metrics = risk_manager.get_risk_metrics()
        print(f"Risk Metrics: {json.dumps(metrics, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        risk_manager.stop_monitoring()

if __name__ == "__main__":
    main()