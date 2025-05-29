# 🚀 Hybrid Retail HFT Bot for MetaTrader 5

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![MQL5](https://img.shields.io/badge/MQL5-Compatible-green.svg)](https://www.mql5.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](TEST_REPORT.md)

A sophisticated, AI-driven high-frequency algorithmic trading bot for MetaTrader 5 (MT5), designed to bridge institutional HFT principles with retail trading constraints. Achieve ultra-low latency execution, high-volume trading, and institutional-grade risk management.

## 🎯 Objectives

- **High-Volume Trading**: Target thousands of trades per day (aiming for 50,000 where feasible)
- **Ultra-Low Latency**: Optimize for millisecond-range execution (10-50ms)
- **AI-Driven Decisions**: Leverage machine learning for prediction and execution optimization
- **Risk Management**: Comprehensive safeguards and automated risk controls
- **Scalability**: Modular architecture for easy expansion and optimization

## 🏗️ Architecture

```
hft_bot/
├── mql5/                   # MetaTrader 5 Expert Advisors and Scripts
│   ├── experts/           # Main trading EAs
│   ├── indicators/        # Custom indicators
│   ├── libraries/         # Shared MQL5 libraries
│   └── scripts/           # Utility scripts
├── python/                # Python AI/ML components
│   ├── models/           # ML model definitions
│   ├── data/             # Data processing pipeline
│   ├── execution/        # Order execution optimization
│   ├── risk/             # Risk management
│   └── monitoring/       # Performance monitoring
├── config/               # Configuration files
├── data/                 # Historical and real-time data
├── monitoring/           # Monitoring dashboards
├── infrastructure/       # VPS setup and optimization
└── tests/               # Testing framework
```

## 🚀 Key Features

### 1. Ultra-Low Latency Infrastructure
- VPS optimization for sub-millisecond broker connectivity
- High-performance hardware specifications
- Network optimization with 10GbE interfaces

### 2. AI/ML Integration
- Deep Learning models (LSTM, Transformers) for price prediction
- Reinforcement Learning for execution optimization
- Real-time anomaly detection
- ONNX model deployment in MQL5

### 3. Advanced Data Pipeline
- Real-time tick data acquisition
- Level 2 order book analysis
- Feature engineering for HFT
- ZeroMQ communication between Python and MQL5

### 4. Comprehensive Risk Management
- Dynamic position sizing
- Automated kill switches
- Circuit breakers for abnormal conditions
- Real-time monitoring and alerting

### 5. Strategy Framework
- Scalping strategies
- Mean reversion algorithms
- Statistical arbitrage
- Market microstructure analysis

## 📊 Performance Targets

- **Execution Latency**: <50ms average
- **Trade Volume**: 10,000-50,000 trades/day
- **Win Rate**: >60% with positive expectancy
- **Sharpe Ratio**: >2.0
- **Maximum Drawdown**: <5%

## 🛠️ Installation & Setup

See detailed setup instructions in `/infrastructure/README.md`

## 📈 Monitoring

Real-time performance monitoring via web dashboard accessible at:
- Development: http://localhost:12000
- Production: Custom VPS deployment

## ⚠️ Risk Disclaimer

This is a high-frequency trading system designed for experienced traders. High-frequency trading involves significant risks including:
- Rapid capital loss
- Technology failures
- Market volatility
- Regulatory changes

Always test thoroughly in demo environments before live deployment.

## 📄 License

MIT License - See LICENSE file for details