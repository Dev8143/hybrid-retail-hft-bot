"""
Real-time HFT Monitoring Dashboard
Web-based dashboard for monitoring HFT bot performance and risk metrics
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from collections import deque
import logging
import zmq
from typing import Dict, List, Any, Optional
import sqlite3
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and stores monitoring data"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.init_database()
        
        # In-memory data for real-time display
        self.tick_data = deque(maxlen=10000)
        self.trade_data = deque(maxlen=5000)
        self.performance_data = deque(maxlen=1000)
        self.risk_data = deque(maxlen=1000)
        self.latency_data = deque(maxlen=1000)
        
        # ZMQ connection for real-time data
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        self.collector_thread = None
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                symbol TEXT,
                bid REAL,
                ask REAL,
                spread REAL,
                volume INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                symbol TEXT,
                side TEXT,
                size REAL,
                price REAL,
                pnl REAL,
                latency REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                equity REAL,
                balance REAL,
                daily_pnl REAL,
                total_trades INTEGER,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                total_exposure REAL,
                position_count INTEGER,
                var_1d REAL,
                correlation_risk REAL,
                volatility REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def start_collection(self, zmq_address: str = "tcp://localhost:5557"):
        """Start data collection from ZMQ"""
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(zmq_address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        
        logger.info(f"Data collection started from {zmq_address}")
        
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)
        if self.socket:
            self.socket.close()
        self.context.term()
        
    def _collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                if self.socket.poll(timeout=100):  # 100ms timeout
                    message = self.socket.recv_string(zmq.NOBLOCK)
                    self._process_message(message)
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                time.sleep(0.1)
                
    def _process_message(self, message: str):
        """Process incoming ZMQ message"""
        try:
            parts = message.split(' ', 1)
            if len(parts) != 2:
                return
                
            msg_type, data_str = parts
            data = json.loads(data_str)
            
            timestamp = data.get('timestamp', time.time())
            
            if msg_type == 'tick':
                self._process_tick(data, timestamp)
            elif msg_type == 'trade':
                self._process_trade(data, timestamp)
            elif msg_type == 'performance':
                self._process_performance(data, timestamp)
            elif msg_type == 'risk':
                self._process_risk(data, timestamp)
            elif msg_type == 'monitoring':
                self._process_monitoring(data, timestamp)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def _process_tick(self, data: Dict, timestamp: float):
        """Process tick data"""
        tick_record = {
            'timestamp': timestamp,
            'symbol': data.get('symbol', 'UNKNOWN'),
            'bid': data.get('bid', 0.0),
            'ask': data.get('ask', 0.0),
            'spread': data.get('spread', 0.0),
            'volume': data.get('volume', 0)
        }
        self.tick_data.append(tick_record)
        
    def _process_trade(self, data: Dict, timestamp: float):
        """Process trade data"""
        trade_record = {
            'timestamp': timestamp,
            'symbol': data.get('symbol', 'UNKNOWN'),
            'side': data.get('side', 'BUY'),
            'size': data.get('size', 0.0),
            'price': data.get('price', 0.0),
            'pnl': data.get('pnl', 0.0),
            'latency': data.get('latency', 0.0)
        }
        self.trade_data.append(trade_record)
        
        # Add latency data
        if trade_record['latency'] > 0:
            self.latency_data.append({
                'timestamp': timestamp,
                'latency': trade_record['latency']
            })
        
    def _process_performance(self, data: Dict, timestamp: float):
        """Process performance data"""
        perf_record = {
            'timestamp': timestamp,
            'equity': data.get('equity', 0.0),
            'balance': data.get('balance', 0.0),
            'daily_pnl': data.get('daily_pnl', 0.0),
            'total_trades': data.get('total_trades', 0),
            'win_rate': data.get('win_rate', 0.0),
            'sharpe_ratio': data.get('sharpe_ratio', 0.0),
            'max_drawdown': data.get('max_drawdown', 0.0)
        }
        self.performance_data.append(perf_record)
        
    def _process_risk(self, data: Dict, timestamp: float):
        """Process risk data"""
        risk_record = {
            'timestamp': timestamp,
            'total_exposure': data.get('total_exposure', 0.0),
            'position_count': data.get('position_count', 0),
            'var_1d': data.get('var_1d', 0.0),
            'correlation_risk': data.get('correlation_risk', 0.0),
            'volatility': data.get('volatility', 0.0)
        }
        self.risk_data.append(risk_record)
        
    def _process_monitoring(self, data: Dict, timestamp: float):
        """Process general monitoring data"""
        # Update multiple data sources from monitoring message
        if 'equity' in data:
            self._process_performance(data, timestamp)
        if 'total_exposure' in data:
            self._process_risk(data, timestamp)
            
    def get_recent_data(self, data_type: str, minutes: int = 60) -> List[Dict]:
        """Get recent data of specified type"""
        cutoff_time = time.time() - (minutes * 60)
        
        if data_type == 'ticks':
            return [d for d in self.tick_data if d['timestamp'] >= cutoff_time]
        elif data_type == 'trades':
            return [d for d in self.trade_data if d['timestamp'] >= cutoff_time]
        elif data_type == 'performance':
            return [d for d in self.performance_data if d['timestamp'] >= cutoff_time]
        elif data_type == 'risk':
            return [d for d in self.risk_data if d['timestamp'] >= cutoff_time]
        elif data_type == 'latency':
            return [d for d in self.latency_data if d['timestamp'] >= cutoff_time]
        else:
            return []

class HFTDashboard:
    """Main HFT monitoring dashboard"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("HFT Bot Monitoring Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.Div(id='status-indicator', 
                        style={'textAlign': 'center', 'fontSize': '18px'})
            ], style={'marginBottom': '20px'}),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            ),
            
            # Main content tabs
            dcc.Tabs(id='main-tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Performance', value='performance'),
                dcc.Tab(label='Risk Management', value='risk'),
                dcc.Tab(label='Latency', value='latency'),
                dcc.Tab(label='Positions', value='positions'),
                dcc.Tab(label='System Health', value='system')
            ]),
            
            # Tab content
            html.Div(id='tab-content')
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('main-tabs', 'value'),
            Input('interval-component', 'n_intervals')
        )
        def update_tab_content(active_tab, n_intervals):
            if active_tab == 'overview':
                return self.create_overview_tab()
            elif active_tab == 'performance':
                return self.create_performance_tab()
            elif active_tab == 'risk':
                return self.create_risk_tab()
            elif active_tab == 'latency':
                return self.create_latency_tab()
            elif active_tab == 'positions':
                return self.create_positions_tab()
            elif active_tab == 'system':
                return self.create_system_tab()
            else:
                return html.Div("Select a tab")
                
        @self.app.callback(
            Output('status-indicator', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_status(n_intervals):
            # Check if data is flowing
            recent_data = self.data_collector.get_recent_data('performance', 1)
            if recent_data:
                status = "🟢 LIVE"
                color = "green"
            else:
                status = "🔴 NO DATA"
                color = "red"
                
            return html.Span(status, style={'color': color, 'fontWeight': 'bold'})
    
    def create_overview_tab(self):
        """Create overview tab content"""
        # Get recent data
        perf_data = self.data_collector.get_recent_data('performance', 60)
        trade_data = self.data_collector.get_recent_data('trades', 60)
        risk_data = self.data_collector.get_recent_data('risk', 60)
        
        # Calculate summary metrics
        current_equity = perf_data[-1]['equity'] if perf_data else 0
        daily_pnl = perf_data[-1]['daily_pnl'] if perf_data else 0
        total_trades = len(trade_data)
        current_positions = risk_data[-1]['position_count'] if risk_data else 0
        
        return html.Div([
            # Key metrics cards
            html.Div([
                html.Div([
                    html.H3(f"${current_equity:,.2f}", style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("Current Equity", style={'margin': '0', 'color': '#7f8c8d'})
                ], className='metric-card', style=self.card_style()),
                
                html.Div([
                    html.H3(f"${daily_pnl:+,.2f}", 
                           style={'margin': '0', 'color': 'green' if daily_pnl >= 0 else 'red'}),
                    html.P("Daily P&L", style={'margin': '0', 'color': '#7f8c8d'})
                ], className='metric-card', style=self.card_style()),
                
                html.Div([
                    html.H3(f"{total_trades:,}", style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("Trades (1h)", style={'margin': '0', 'color': '#7f8c8d'})
                ], className='metric-card', style=self.card_style()),
                
                html.Div([
                    html.H3(f"{current_positions}", style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("Open Positions", style={'margin': '0', 'color': '#7f8c8d'})
                ], className='metric-card', style=self.card_style())
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
            
            # Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='equity-chart', figure=self.create_equity_chart())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='trades-chart', figure=self.create_trades_chart())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def create_performance_tab(self):
        """Create performance tab content"""
        return html.Div([
            html.H3("Performance Metrics"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='pnl-chart', figure=self.create_pnl_chart())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='drawdown-chart', figure=self.create_drawdown_chart())
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='sharpe-chart', figure=self.create_sharpe_chart())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='winrate-chart', figure=self.create_winrate_chart())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def create_risk_tab(self):
        """Create risk management tab content"""
        return html.Div([
            html.H3("Risk Management"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='exposure-chart', figure=self.create_exposure_chart())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='var-chart', figure=self.create_var_chart())
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='correlation-chart', figure=self.create_correlation_chart())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='volatility-chart', figure=self.create_volatility_chart())
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    def create_latency_tab(self):
        """Create latency monitoring tab content"""
        return html.Div([
            html.H3("Latency Monitoring"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='latency-timeseries', figure=self.create_latency_timeseries())
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='latency-histogram', figure=self.create_latency_histogram())
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='latency-percentiles', figure=self.create_latency_percentiles())
                ], style={'width': '100%'})
            ])
        ])
    
    def create_positions_tab(self):
        """Create positions tab content"""
        return html.Div([
            html.H3("Current Positions"),
            html.Div(id='positions-table')
        ])
    
    def create_system_tab(self):
        """Create system health tab content"""
        return html.Div([
            html.H3("System Health"),
            html.Div(id='system-metrics')
        ])
    
    def card_style(self):
        """Style for metric cards"""
        return {
            'backgroundColor': '#ecf0f1',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'minWidth': '150px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }
    
    def create_equity_chart(self):
        """Create equity curve chart"""
        data = self.data_collector.get_recent_data('performance', 240)  # 4 hours
        
        if not data:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity ($)',
            height=400
        )
        
        return fig
    
    def create_trades_chart(self):
        """Create trades per minute chart"""
        data = self.data_collector.get_recent_data('trades', 60)
        
        if not data:
            return go.Figure().add_annotation(text="No data available", 
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['minute'] = df['datetime'].dt.floor('min')
        
        trades_per_minute = df.groupby('minute').size().reset_index(name='count')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trades_per_minute['minute'],
            y=trades_per_minute['count'],
            name='Trades per Minute'
        ))
        
        fig.update_layout(
            title='Trading Activity',
            xaxis_title='Time',
            yaxis_title='Trades per Minute',
            height=400
        )
        
        return fig
    
    def create_pnl_chart(self):
        """Create P&L chart"""
        data = self.data_collector.get_recent_data('performance', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['daily_pnl'],
            mode='lines',
            name='Daily P&L',
            line=dict(color='green' if df['daily_pnl'].iloc[-1] >= 0 else 'red')
        ))
        
        fig.update_layout(
            title='Daily P&L',
            xaxis_title='Time',
            yaxis_title='P&L ($)',
            height=300
        )
        
        return fig
    
    def create_drawdown_chart(self):
        """Create drawdown chart"""
        data = self.data_collector.get_recent_data('performance', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['max_drawdown'],
            mode='lines',
            name='Max Drawdown',
            line=dict(color='red'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title='Maximum Drawdown',
            xaxis_title='Time',
            yaxis_title='Drawdown (%)',
            height=300
        )
        
        return fig
    
    def create_sharpe_chart(self):
        """Create Sharpe ratio chart"""
        data = self.data_collector.get_recent_data('performance', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['sharpe_ratio'],
            mode='lines',
            name='Sharpe Ratio',
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title='Sharpe Ratio',
            xaxis_title='Time',
            yaxis_title='Sharpe Ratio',
            height=300
        )
        
        return fig
    
    def create_winrate_chart(self):
        """Create win rate chart"""
        data = self.data_collector.get_recent_data('performance', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['win_rate'] * 100,
            mode='lines',
            name='Win Rate',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title='Win Rate',
            xaxis_title='Time',
            yaxis_title='Win Rate (%)',
            height=300
        )
        
        return fig
    
    def create_exposure_chart(self):
        """Create exposure chart"""
        data = self.data_collector.get_recent_data('risk', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['total_exposure'],
            mode='lines',
            name='Total Exposure',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Total Exposure',
            xaxis_title='Time',
            yaxis_title='Exposure ($)',
            height=300
        )
        
        return fig
    
    def create_var_chart(self):
        """Create VaR chart"""
        data = self.data_collector.get_recent_data('risk', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['var_1d'],
            mode='lines',
            name='1-Day VaR',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Value at Risk (1-Day)',
            xaxis_title='Time',
            yaxis_title='VaR ($)',
            height=300
        )
        
        return fig
    
    def create_correlation_chart(self):
        """Create correlation chart"""
        # Placeholder for correlation visualization
        fig = go.Figure()
        fig.add_annotation(text="Correlation matrix visualization", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        
        fig.update_layout(
            title='Position Correlations',
            height=300
        )
        
        return fig
    
    def create_volatility_chart(self):
        """Create volatility chart"""
        data = self.data_collector.get_recent_data('risk', 240)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['volatility'] * 100,
            mode='lines',
            name='Portfolio Volatility',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title='Portfolio Volatility',
            xaxis_title='Time',
            yaxis_title='Volatility (%)',
            height=300
        )
        
        return fig
    
    def create_latency_timeseries(self):
        """Create latency time series chart"""
        data = self.data_collector.get_recent_data('latency', 60)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['latency'],
            mode='markers',
            name='Latency',
            marker=dict(size=4, color='red')
        ))
        
        fig.update_layout(
            title='Execution Latency',
            xaxis_title='Time',
            yaxis_title='Latency (ms)',
            height=300
        )
        
        return fig
    
    def create_latency_histogram(self):
        """Create latency histogram"""
        data = self.data_collector.get_recent_data('latency', 60)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['latency'],
            nbinsx=30,
            name='Latency Distribution'
        ))
        
        fig.update_layout(
            title='Latency Distribution',
            xaxis_title='Latency (ms)',
            yaxis_title='Frequency',
            height=300
        )
        
        return fig
    
    def create_latency_percentiles(self):
        """Create latency percentiles chart"""
        data = self.data_collector.get_recent_data('latency', 60)
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        latencies = df['latency'].values
        
        percentiles = [50, 75, 90, 95, 99]
        values = [np.percentile(latencies, p) for p in percentiles]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'P{p}' for p in percentiles],
            y=values,
            name='Latency Percentiles'
        ))
        
        fig.update_layout(
            title='Latency Percentiles',
            xaxis_title='Percentile',
            yaxis_title='Latency (ms)',
            height=300
        )
        
        return fig
    
    def run(self, host: str = '0.0.0.0', port: int = 12000, debug: bool = False):
        """Run the dashboard"""
        logger.info(f"Starting HFT Dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the dashboard"""
    
    # Create data collector
    data_collector = DataCollector()
    
    # Start data collection (comment out if no ZMQ source available)
    # data_collector.start_collection()
    
    # Generate some sample data for demonstration
    import random
    current_time = time.time()
    
    for i in range(100):
        timestamp = current_time - (100 - i) * 60  # 1 minute intervals
        
        # Sample performance data
        data_collector.performance_data.append({
            'timestamp': timestamp,
            'equity': 10000 + random.gauss(0, 100) + i * 10,
            'balance': 10000 + random.gauss(0, 50) + i * 5,
            'daily_pnl': random.gauss(0, 50),
            'total_trades': i * 10,
            'win_rate': 0.6 + random.gauss(0, 0.1),
            'sharpe_ratio': 1.5 + random.gauss(0, 0.3),
            'max_drawdown': abs(random.gauss(0, 2))
        })
        
        # Sample risk data
        data_collector.risk_data.append({
            'timestamp': timestamp,
            'total_exposure': random.uniform(1000, 5000),
            'position_count': random.randint(1, 10),
            'var_1d': random.uniform(50, 200),
            'correlation_risk': random.uniform(0, 0.8),
            'volatility': random.uniform(0.01, 0.03)
        })
        
        # Sample latency data
        if i % 5 == 0:  # Every 5th point
            data_collector.latency_data.append({
                'timestamp': timestamp,
                'latency': random.exponential(20)
            })
    
    # Create and run dashboard
    dashboard = HFTDashboard(data_collector)
    
    try:
        dashboard.run(host='0.0.0.0', port=12000, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down dashboard...")
    finally:
        data_collector.stop_collection()

if __name__ == "__main__":
    main()