//+------------------------------------------------------------------+
//|                                                 HFT_Master_EA.mq5 |
//|                                    Hybrid Retail HFT Trading Bot |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "HFT Bot Team"
#property link      ""
#property version   "1.00"
#property description "AI-Driven High-Frequency Trading Expert Advisor"

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>
#include <Math\Stat\Math.mqh>

//--- Input parameters
input group "=== TRADING PARAMETERS ==="
input double   InpLotSize = 0.01;              // Lot size
input int      InpMagicNumber = 123456;        // Magic number
input int      InpMaxPositions = 10;           // Maximum concurrent positions
input double   InpMaxRiskPercent = 2.0;        // Maximum risk per trade (%)
input int      InpMaxTradesPerDay = 50000;     // Maximum trades per day

input group "=== AI/ML PARAMETERS ==="
input string   InpModelPath = "models/hft_model.onnx";  // ONNX model path
input bool     InpUseAI = true;                // Enable AI predictions
input double   InpAIConfidenceThreshold = 0.7; // Minimum AI confidence
input int      InpPredictionHorizon = 5;       // Prediction horizon (ticks)

input group "=== LATENCY OPTIMIZATION ==="
input bool     InpOptimizeLatency = true;      // Enable latency optimization
input int      InpMaxLatencyMs = 50;           // Maximum acceptable latency (ms)
input bool     InpUseAsyncExecution = true;    // Use asynchronous execution

input group "=== RISK MANAGEMENT ==="
input double   InpMaxDrawdownPercent = 5.0;    // Maximum drawdown (%)
input double   InpDailyLossLimit = 1000.0;     // Daily loss limit ($)
input bool     InpUseKillSwitch = true;        // Enable emergency kill switch
input int      InpStopLossPoints = 100;        // Stop loss in points
input int      InpTakeProfitPoints = 150;      // Take profit in points

input group "=== DATA FEED ==="
input bool     InpUseExternalData = true;      // Use external data feed
input string   InpZMQAddress = "tcp://localhost:5555"; // ZeroMQ address
input int      InpDataUpdateFrequency = 1;     // Data update frequency (ms)

//--- Global variables
CTrade         trade;
CSymbolInfo    symbolInfo;
CPositionInfo  positionInfo;
COrderInfo     orderInfo;

// Performance tracking
datetime       lastTradeTime;
int            tradesCount;
double         dailyPnL;
double         maxDrawdown;
double         peakEquity;
ulong          lastTickTime;
double         avgLatency;
int            latencyCount;

// Market data structure
struct MarketData {
    double bid;
    double ask;
    double spread;
    long volume;
    datetime time;
    double features[50];  // Feature vector for ML
};

MarketData currentMarket;
MarketData marketHistory[1000];  // Circular buffer
int historyIndex = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Initializing HFT Master EA...");
    
    // Initialize trading objects
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(Symbol());
    
    // Initialize symbol info
    if (!symbolInfo.Name(Symbol())) {
        Print("Failed to initialize symbol info");
        return INIT_FAILED;
    }
    
    // Reset daily counters
    ResetDailyCounters();
    
    // Setup timer for high-frequency updates
    if (InpDataUpdateFrequency > 0) {
        EventSetMillisecondTimer(InpDataUpdateFrequency);
    }
    
    Print("HFT Master EA initialized successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    Print("Deinitializing HFT Master EA...");
    
    // Emergency position closure if needed
    if (InpUseKillSwitch && reason == REASON_REMOVE) {
        CloseAllPositions("EA Removal");
    }
    
    EventKillTimer();
    Print("HFT Master EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    ulong tickStart = GetMicrosecondCount();
    
    // Update market data
    UpdateMarketData();
    
    // Check daily trade limit
    if (tradesCount >= InpMaxTradesPerDay) {
        return;
    }
    
    // Process market data and generate signals
    ProcessMarketData();
    
    // Execute trading logic
    ExecuteTradingLogic();
    
    // Update performance metrics
    UpdatePerformanceMetrics();
    
    // Calculate and track latency
    ulong tickEnd = GetMicrosecondCount();
    double latency = (tickEnd - tickStart) / 1000.0; // Convert to milliseconds
    UpdateLatencyMetrics(latency);
    
    // Check latency threshold
    if (InpOptimizeLatency && latency > InpMaxLatencyMs) {
        Print("Warning: High latency detected: ", latency, "ms");
    }
}

//+------------------------------------------------------------------+
//| Timer function for high-frequency updates                       |
//+------------------------------------------------------------------+
void OnTimer() {
    // Update monitoring data
    SendMonitoringData();
}

//+------------------------------------------------------------------+
//| Update market data structure                                    |
//+------------------------------------------------------------------+
void UpdateMarketData() {
    MqlTick tick;
    if (!SymbolInfoTick(Symbol(), tick)) {
        return;
    }
    
    currentMarket.bid = tick.bid;
    currentMarket.ask = tick.ask;
    currentMarket.spread = tick.ask - tick.bid;
    currentMarket.volume = tick.volume;
    currentMarket.time = tick.time;
    
    // Store in history buffer
    marketHistory[historyIndex] = currentMarket;
    historyIndex = (historyIndex + 1) % ArraySize(marketHistory);
    
    lastTickTime = tick.time_msc;
}

//+------------------------------------------------------------------+
//| Process market data and extract features                        |
//+------------------------------------------------------------------+
void ProcessMarketData() {
    // Extract features for ML model
    CalculateAdvancedFeatures();
}

//+------------------------------------------------------------------+
//| Calculate advanced features for ML                              |
//+------------------------------------------------------------------+
void CalculateAdvancedFeatures() {
    // Price momentum features
    currentMarket.features[0] = CalculateMomentum(5);
    currentMarket.features[1] = CalculateMomentum(10);
    currentMarket.features[2] = CalculateMomentum(20);
    
    // Volatility features
    currentMarket.features[3] = CalculateVolatility(10);
    currentMarket.features[4] = CalculateVolatility(20);
    
    // Spread features
    currentMarket.features[5] = currentMarket.spread / currentMarket.bid;
    currentMarket.features[6] = CalculateSpreadMomentum(5);
    
    // Volume features
    currentMarket.features[7] = (double)currentMarket.volume;
    currentMarket.features[8] = CalculateVolumeRatio(10);
    
    // Time-based features
    MqlDateTime dt;
    TimeToStruct(currentMarket.time, dt);
    currentMarket.features[9] = dt.hour;
    currentMarket.features[10] = dt.min;
    currentMarket.features[11] = dt.sec;
}

//+------------------------------------------------------------------+
//| Main trading logic execution                                    |
//+------------------------------------------------------------------+
void ExecuteTradingLogic() {
    // Generate trading signals
    int signal = GenerateTradingSignal();
    
    if (signal != 0) {
        ExecuteTradeSignal(signal, 0.8);
    }
    
    // Manage existing positions
    ManagePositions();
}

//+------------------------------------------------------------------+
//| Generate trading signal                                          |
//+------------------------------------------------------------------+
int GenerateTradingSignal() {
    // Combine multiple signal sources
    int signal = 0;
    
    // Mean reversion signal
    double meanReversionSignal = CalculateMeanReversionSignal();
    if (MathAbs(meanReversionSignal) > 0.3) {
        signal += (meanReversionSignal > 0) ? 1 : -1;
    }
    
    // Momentum signal
    double momentumSignal = CalculateMomentumSignal();
    if (MathAbs(momentumSignal) > 0.3) {
        signal += (momentumSignal > 0) ? 1 : -1;
    }
    
    // Microstructure signal
    double microSignal = CalculateMicrostructureSignal();
    if (MathAbs(microSignal) > 0.2) {
        signal += (microSignal > 0) ? 1 : -1;
    }
    
    // Normalize signal
    if (signal > 1) return 1;
    if (signal < -1) return -1;
    
    return 0;
}

//+------------------------------------------------------------------+
//| Execute trade signal                                             |
//+------------------------------------------------------------------+
void ExecuteTradeSignal(int signal, double confidence) {
    if (signal == 0) return;
    
    // Check position limits
    if (PositionsTotal() >= InpMaxPositions) {
        return;
    }
    
    // Calculate position size
    double lotSize = CalculatePositionSize(confidence);
    if (lotSize < symbolInfo.LotsMin()) {
        return;
    }
    
    // Calculate stop loss and take profit
    double sl = 0, tp = 0;
    CalculateStopLevels(signal, sl, tp);
    
    // Execute trade
    bool result = false;
    if (signal > 0) {
        result = trade.Buy(lotSize, Symbol(), 0, sl, tp, "HFT_Buy");
    } else {
        result = trade.Sell(lotSize, Symbol(), 0, sl, tp, "HFT_Sell");
    }
    
    if (result) {
        tradesCount++;
        lastTradeTime = TimeCurrent();
        Print("Trade executed: ", (signal > 0) ? "BUY" : "SELL", 
              " Lot: ", lotSize, " Confidence: ", confidence);
    }
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk and confidence            |
//+------------------------------------------------------------------+
double CalculatePositionSize(double confidence) {
    double baseSize = InpLotSize;
    
    // Adjust size based on confidence
    if (confidence > 0) {
        baseSize *= (0.5 + confidence * 0.5); // Scale from 0.5x to 1.0x
    }
    
    // Apply risk management
    double maxRiskAmount = AccountInfoDouble(ACCOUNT_EQUITY) * InpMaxRiskPercent / 100.0;
    double riskPerPoint = symbolInfo.TickValue() * baseSize / symbolInfo.TickSize();
    double maxLotsByRisk = maxRiskAmount / (InpStopLossPoints * riskPerPoint);
    
    baseSize = MathMin(baseSize, maxLotsByRisk);
    
    // Ensure minimum lot size
    baseSize = MathMax(baseSize, symbolInfo.LotsMin());
    
    // Ensure maximum lot size
    baseSize = MathMin(baseSize, symbolInfo.LotsMax());
    
    return NormalizeDouble(baseSize, 2);
}

//+------------------------------------------------------------------+
//| Calculate stop loss and take profit levels                      |
//+------------------------------------------------------------------+
void CalculateStopLevels(int signal, double &sl, double &tp) {
    double point = symbolInfo.Point();
    
    if (signal > 0) { // Buy
        sl = currentMarket.bid - InpStopLossPoints * point;
        tp = currentMarket.ask + InpTakeProfitPoints * point;
    } else { // Sell
        sl = currentMarket.ask + InpStopLossPoints * point;
        tp = currentMarket.bid - InpTakeProfitPoints * point;
    }
    
    // Adjust for minimum stop level
    double minStopLevel = symbolInfo.StopsLevel() * point;
    if (signal > 0) {
        sl = MathMin(sl, currentMarket.bid - minStopLevel);
        tp = MathMax(tp, currentMarket.ask + minStopLevel);
    } else {
        sl = MathMax(sl, currentMarket.ask + minStopLevel);
        tp = MathMin(tp, currentMarket.bid - minStopLevel);
    }
}

//+------------------------------------------------------------------+
//| Manage existing positions                                        |
//+------------------------------------------------------------------+
void ManagePositions() {
    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        if (positionInfo.SelectByIndex(i)) {
            if (positionInfo.Symbol() == Symbol() && 
                positionInfo.Magic() == InpMagicNumber) {
                
                // Check for time-based exit (HFT positions should be short-lived)
                if (TimeCurrent() - positionInfo.Time() > 300) { // 5 minutes max
                    trade.PositionClose(positionInfo.Ticket());
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate momentum indicator                                     |
//+------------------------------------------------------------------+
double CalculateMomentum(int period) {
    if (historyIndex < period) return 0.0;
    
    int startIdx = (historyIndex - period + ArraySize(marketHistory)) % ArraySize(marketHistory);
    double startPrice = (marketHistory[startIdx].bid + marketHistory[startIdx].ask) / 2.0;
    double currentPrice = (currentMarket.bid + currentMarket.ask) / 2.0;
    
    return (currentPrice - startPrice) / startPrice;
}

//+------------------------------------------------------------------+
//| Calculate volatility                                             |
//+------------------------------------------------------------------+
double CalculateVolatility(int period) {
    if (historyIndex < period) return 0.0;
    
    double sum = 0.0;
    double sumSq = 0.0;
    
    for (int i = 0; i < period; i++) {
        int idx = (historyIndex - i - 1 + ArraySize(marketHistory)) % ArraySize(marketHistory);
        double price = (marketHistory[idx].bid + marketHistory[idx].ask) / 2.0;
        sum += price;
        sumSq += price * price;
    }
    
    double mean = sum / period;
    double variance = (sumSq / period) - (mean * mean);
    
    return MathSqrt(variance);
}

//+------------------------------------------------------------------+
//| Calculate mean reversion signal                                 |
//+------------------------------------------------------------------+
double CalculateMeanReversionSignal() {
    double sma20 = CalculateSMA(20);
    double currentPrice = (currentMarket.bid + currentMarket.ask) / 2.0;
    
    if (sma20 == 0.0) return 0.0;
    
    double deviation = (currentPrice - sma20) / sma20;
    
    // Return opposite signal for mean reversion
    return -deviation * 2.0; // Amplify signal
}

//+------------------------------------------------------------------+
//| Calculate momentum signal                                        |
//+------------------------------------------------------------------+
double CalculateMomentumSignal() {
    double shortMom = CalculateMomentum(5);
    double longMom = CalculateMomentum(20);
    
    return shortMom - longMom;
}

//+------------------------------------------------------------------+
//| Calculate microstructure signal                                 |
//+------------------------------------------------------------------+
double CalculateMicrostructureSignal() {
    double spreadRatio = currentMarket.spread / ((currentMarket.bid + currentMarket.ask) / 2.0);
    double avgSpread = CalculateAverageSpread(10);
    
    if (avgSpread == 0.0) return 0.0;
    
    // Signal based on spread anomalies
    return (avgSpread - spreadRatio) / avgSpread;
}

//+------------------------------------------------------------------+
//| Calculate Simple Moving Average                                 |
//+------------------------------------------------------------------+
double CalculateSMA(int period) {
    if (historyIndex < period) return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        int idx = (historyIndex - i - 1 + ArraySize(marketHistory)) % ArraySize(marketHistory);
        sum += (marketHistory[idx].bid + marketHistory[idx].ask) / 2.0;
    }
    
    return sum / period;
}

//+------------------------------------------------------------------+
//| Calculate average spread                                         |
//+------------------------------------------------------------------+
double CalculateAverageSpread(int period) {
    if (historyIndex < period) return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < period; i++) {
        int idx = (historyIndex - i - 1 + ArraySize(marketHistory)) % ArraySize(marketHistory);
        sum += marketHistory[idx].spread;
    }
    
    return sum / period;
}

//+------------------------------------------------------------------+
//| Calculate spread momentum                                        |
//+------------------------------------------------------------------+
double CalculateSpreadMomentum(int period) {
    if (historyIndex < period) return 0.0;
    
    int startIdx = (historyIndex - period + ArraySize(marketHistory)) % ArraySize(marketHistory);
    double startSpread = marketHistory[startIdx].spread;
    
    if (startSpread == 0.0) return 0.0;
    
    return (currentMarket.spread - startSpread) / startSpread;
}

//+------------------------------------------------------------------+
//| Calculate volume ratio                                           |
//+------------------------------------------------------------------+
double CalculateVolumeRatio(int period) {
    if (historyIndex < period) return 1.0;
    
    double avgVolume = 0.0;
    for (int i = 0; i < period; i++) {
        int idx = (historyIndex - i - 1 + ArraySize(marketHistory)) % ArraySize(marketHistory);
        avgVolume += (double)marketHistory[idx].volume;
    }
    avgVolume /= period;
    
    if (avgVolume == 0.0) return 1.0;
    
    return (double)currentMarket.volume / avgVolume;
}

//+------------------------------------------------------------------+
//| Send monitoring data                                             |
//+------------------------------------------------------------------+
void SendMonitoringData() {
    // Create monitoring data structure
    string monitoringData = CreateMonitoringJSON();
    Print("Monitoring: ", monitoringData);
}

//+------------------------------------------------------------------+
//| Create monitoring JSON data                                      |
//+------------------------------------------------------------------+
string CreateMonitoringJSON() {
    string json = "{";
    json += "\"timestamp\":" + IntegerToString(TimeCurrent()) + ",";
    json += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
    json += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
    json += "\"trades_count\":" + IntegerToString(tradesCount) + ",";
    json += "\"daily_pnl\":" + DoubleToString(dailyPnL, 2) + ",";
    json += "\"max_drawdown\":" + DoubleToString(maxDrawdown, 2) + ",";
    json += "\"avg_latency\":" + DoubleToString(avgLatency, 2) + ",";
    json += "\"positions\":" + IntegerToString(PositionsTotal()) + ",";
    json += "\"spread\":" + DoubleToString(currentMarket.spread, 5);
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Update performance metrics                                       |
//+------------------------------------------------------------------+
void UpdatePerformanceMetrics() {
    double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Update peak equity and drawdown
    if (currentEquity > peakEquity) {
        peakEquity = currentEquity;
    }
    
    double currentDrawdown = (peakEquity - currentEquity) / peakEquity * 100.0;
    if (currentDrawdown > maxDrawdown) {
        maxDrawdown = currentDrawdown;
    }
    
    // Calculate daily P&L
    static double startDayBalance = 0.0;
    if (TimeDay(TimeCurrent()) != TimeDay(lastTradeTime) || startDayBalance == 0.0) {
        startDayBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        ResetDailyCounters();
    }
    
    dailyPnL = AccountInfoDouble(ACCOUNT_BALANCE) - startDayBalance;
}

//+------------------------------------------------------------------+
//| Update latency metrics                                           |
//+------------------------------------------------------------------+
void UpdateLatencyMetrics(double latency) {
    latencyCount++;
    avgLatency = ((avgLatency * (latencyCount - 1)) + latency) / latencyCount;
}

//+------------------------------------------------------------------+
//| Reset daily counters                                             |
//+------------------------------------------------------------------+
void ResetDailyCounters() {
    tradesCount = 0;
    dailyPnL = 0.0;
    latencyCount = 0;
    avgLatency = 0.0;
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason) {
    Print("Closing all positions: ", reason);
    
    for (int i = PositionsTotal() - 1; i >= 0; i--) {
        if (positionInfo.SelectByIndex(i)) {
            if (positionInfo.Symbol() == Symbol() && 
                positionInfo.Magic() == InpMagicNumber) {
                trade.PositionClose(positionInfo.Ticket());
            }
        }
    }
}

//+------------------------------------------------------------------+