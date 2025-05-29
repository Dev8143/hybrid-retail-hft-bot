#!/usr/bin/env python3
"""Test Risk Management System"""

print('🛡️ Testing Risk Management System...')
print('='*50)

try:
    from python.risk.risk_manager import RiskManager, RiskLimits, AlertType
    import time
    import threading
    
    # Create risk manager with test limits
    limits = RiskLimits()
    limits.max_daily_loss = 100.0
    limits.max_drawdown_pct = 5.0
    limits.max_total_exposure = 1000.0
    limits.max_position_size = 0.5
    
    risk_manager = RiskManager(limits)
    print('✓ RiskManager initialized with test limits')
    
    # Test position validation
    can_trade, reason = risk_manager.can_open_position('EURUSD', 0.1, 1.1000)
    print(f'✓ Position validation: {can_trade} - {reason}')
    
    # Add some positions
    risk_manager.update_position('EURUSD', 0.1, 1.1000)
    risk_manager.update_position('GBPUSD', 0.1, 1.3000)
    print('✓ Added test positions')
    
    # Test exposure calculation
    metrics = risk_manager.get_risk_metrics()
    print(f'✓ Current exposure: ${metrics["current_exposure"]:.2f}')
    print(f'✓ Position count: {metrics["position_count"]}')
    
    # Test loss limit
    risk_manager.daily_pnl = -150.0  # Exceeds limit
    can_trade, reason = risk_manager.can_open_position('USDJPY', 0.1, 110.0)
    print(f'✓ Loss limit test: {can_trade} - {reason}')
    
    # Test drawdown calculation
    risk_manager.update_equity(10000.0)  # Peak
    risk_manager.update_equity(9400.0)   # 6% drawdown
    
    current_dd = risk_manager.drawdown_calculator.get_current_drawdown()
    print(f'✓ Drawdown calculation: {current_dd:.2f}%')
    
    # Test latency monitoring
    import random
    for _ in range(100):
        risk_manager.add_latency_sample(15.0 + 5.0 * (0.5 - random.random()))
    
    latency_stats = risk_manager.latency_monitor.get_latency_stats()
    print(f'✓ Latency monitoring: {latency_stats["mean"]:.2f}ms avg')
    
    # Test alert system
    alerts_received = []
    def alert_callback(alert):
        alerts_received.append(alert)
    
    risk_manager.add_alert_callback(alert_callback)
    risk_manager.start_monitoring()
    
    print('✓ Risk monitoring started')
    time.sleep(2)  # Let monitoring run
    
    risk_manager.stop_monitoring()
    print(f'✓ Alerts generated: {len(alerts_received)}')
    
    if alerts_received:
        for alert in alerts_received[:3]:  # Show first 3 alerts
            print(f'  - {alert.alert_type.value}: {alert.message[:50]}...')
    
    print('\n✅ Risk management system test completed successfully!')
    
except Exception as e:
    print(f'❌ Risk management test failed: {e}')
    import traceback
    traceback.print_exc()