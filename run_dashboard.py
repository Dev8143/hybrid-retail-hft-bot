#!/usr/bin/env python3
"""
Quick start script for HFT Bot Dashboard
Runs the monitoring dashboard with sample data
"""

import sys
import os
import time
import threading
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from python.monitoring.dashboard import HFTDashboard, DataCollector
    print("✓ Successfully imported dashboard components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please run 'python setup.py' first to install dependencies")
    sys.exit(1)

def generate_sample_data(data_collector):
    """Generate sample data for demonstration"""
    print("Generating sample data...")
    
    current_time = time.time()
    
    # Generate historical data
    for i in range(200):
        timestamp = current_time - (200 - i) * 30  # 30 second intervals
        
        # Sample performance data
        equity = 10000 + np.random.normal(0, 50) + i * 5
        daily_pnl = np.random.normal(0, 25)
        
        data_collector.performance_data.append({
            'timestamp': timestamp,
            'equity': equity,
            'balance': equity - daily_pnl,
            'daily_pnl': daily_pnl,
            'total_trades': i * 5,
            'win_rate': 0.6 + np.random.normal(0, 0.05),
            'sharpe_ratio': 1.5 + np.random.normal(0, 0.2),
            'max_drawdown': abs(np.random.normal(0, 1.5))
        })
        
        # Sample risk data
        data_collector.risk_data.append({
            'timestamp': timestamp,
            'total_exposure': np.random.uniform(1000, 5000),
            'position_count': np.random.randint(1, 15),
            'var_1d': np.random.uniform(50, 200),
            'correlation_risk': np.random.uniform(0, 0.8),
            'volatility': np.random.uniform(0.01, 0.03)
        })
        
        # Sample trade data
        if i % 10 == 0:  # Every 10th point
            data_collector.trade_data.append({
                'timestamp': timestamp,
                'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY']),
                'side': np.random.choice(['BUY', 'SELL']),
                'size': 0.01,
                'price': 1.1000 + np.random.normal(0, 0.01),
                'pnl': np.random.normal(0, 10),
                'latency': np.random.exponential(20)
            })
        
        # Sample latency data
        if i % 5 == 0:  # Every 5th point
            data_collector.latency_data.append({
                'timestamp': timestamp,
                'latency': np.random.exponential(15) + 5  # 5-50ms range
            })

def update_live_data(data_collector):
    """Continuously update data to simulate live trading"""
    print("Starting live data simulation...")
    
    while True:
        try:
            current_time = time.time()
            
            # Update performance data
            last_perf = data_collector.performance_data[-1] if data_collector.performance_data else None
            if last_perf:
                new_equity = last_perf['equity'] + np.random.normal(0, 10)
                new_pnl = last_perf['daily_pnl'] + np.random.normal(0, 5)
            else:
                new_equity = 10000
                new_pnl = 0
            
            data_collector.performance_data.append({
                'timestamp': current_time,
                'equity': new_equity,
                'balance': new_equity - new_pnl,
                'daily_pnl': new_pnl,
                'total_trades': len(data_collector.trade_data),
                'win_rate': 0.6 + np.random.normal(0, 0.02),
                'sharpe_ratio': 1.5 + np.random.normal(0, 0.1),
                'max_drawdown': abs(np.random.normal(0, 0.5))
            })
            
            # Update risk data
            data_collector.risk_data.append({
                'timestamp': current_time,
                'total_exposure': np.random.uniform(2000, 4000),
                'position_count': np.random.randint(3, 12),
                'var_1d': np.random.uniform(75, 150),
                'correlation_risk': np.random.uniform(0.2, 0.6),
                'volatility': np.random.uniform(0.015, 0.025)
            })
            
            # Occasionally add new trades
            if np.random.random() < 0.3:  # 30% chance
                data_collector.trade_data.append({
                    'timestamp': current_time,
                    'symbol': np.random.choice(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']),
                    'side': np.random.choice(['BUY', 'SELL']),
                    'size': 0.01,
                    'price': 1.1000 + np.random.normal(0, 0.005),
                    'pnl': np.random.normal(2, 8),  # Slightly positive bias
                    'latency': np.random.exponential(12) + 3
                })
            
            # Add latency sample
            if np.random.random() < 0.5:  # 50% chance
                data_collector.latency_data.append({
                    'timestamp': current_time,
                    'latency': np.random.exponential(10) + 5
                })
            
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            print(f"Error in live data update: {e}")
            time.sleep(5)

def main():
    """Main function to run the dashboard"""
    print("="*50)
    print("HFT BOT DASHBOARD")
    print("="*50)
    
    try:
        # Create data collector
        print("Initializing data collector...")
        data_collector = DataCollector()
        
        # Generate initial sample data
        generate_sample_data(data_collector)
        
        # Create dashboard
        print("Creating dashboard...")
        dashboard = HFTDashboard(data_collector)
        
        # Start live data simulation in background
        live_data_thread = threading.Thread(
            target=update_live_data, 
            args=(data_collector,), 
            daemon=True
        )
        live_data_thread.start()
        
        print("\n" + "="*50)
        print("DASHBOARD STARTING...")
        print("="*50)
        print("Dashboard will be available at:")
        print("🌐 http://localhost:12000")
        print("🌐 https://work-1-hrqmmlfkzkzreepr.prod-runtime.all-hands.dev")
        print("\nPress Ctrl+C to stop")
        print("="*50)
        
        # Run dashboard
        dashboard.run(host='0.0.0.0', port=12000, debug=False)
        
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        print("Make sure all dependencies are installed by running 'python setup.py'")
    finally:
        print("Dashboard stopped.")

if __name__ == "__main__":
    main()