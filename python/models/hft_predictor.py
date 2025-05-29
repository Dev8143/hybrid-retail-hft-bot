"""
HFT Predictor Model
AI/ML models for high-frequency trading prediction and execution optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import onnx
import onnxruntime as ort
from typing import Tuple, List, Optional
import logging
import joblib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    """LSTM-based price prediction model for HFT"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Confidence estimation
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        last_output = attn_out[:, -1, :]
        
        # Prediction
        prediction = self.fc_layers(last_output)
        
        # Confidence
        confidence = self.confidence_layer(last_output)
        
        return prediction, confidence

class TransformerPredictor(nn.Module):
    """Transformer-based model for market microstructure prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, output_size: int = 1, dropout: float = 0.1):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
            nn.Tanh()
        )
        
        self.confidence_layer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer forward pass
        transformer_out = self.transformer(x)
        
        # Use the last output
        last_output = transformer_out[:, -1, :]
        
        # Prediction and confidence
        prediction = self.output_layer(last_output)
        confidence = self.confidence_layer(last_output)
        
        return prediction, confidence

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)

class HFTModelTrainer:
    """Main trainer class for HFT models"""
    
    def __init__(self, model_type: str = 'lstm', device: str = 'auto'):
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_columns = None
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for training"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def prepare_data(self, data: pd.DataFrame, sequence_length: int = 50, 
                    target_column: str = 'price_change', test_size: float = 0.2) -> Tuple:
        """Prepare data for training"""
        logger.info(f"Preparing data with {len(data)} samples")
        
        # Store feature columns
        feature_cols = [col for col in data.columns if col != target_column]
        self.feature_columns = feature_cols
        
        # Extract features and targets
        features = data[feature_cols].values
        targets = data[target_column].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, targets, sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Don't shuffle time series
        )
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(sequence_length, len(features)):
            X.append(features[i-sequence_length:i])
            y.append(targets[i])
            
        return np.array(X), np.array(y)
    
    def create_model(self, input_size: int, **kwargs) -> nn.Module:
        """Create the specified model"""
        if self.model_type == 'lstm':
            model = LSTMPredictor(input_size, **kwargs)
        elif self.model_type == 'transformer':
            model = TransformerPredictor(input_size, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model.to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 64, 
              learning_rate: float = 0.001, patience: int = 10) -> dict:
        """Train the model"""
        
        # Create model
        input_size = X_train.shape[-1]
        self.model = self.create_model(input_size)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions, confidence = self.model(batch_X)
                
                # Combined loss: prediction + confidence regularization
                pred_loss = criterion(predictions.squeeze(), batch_y)
                conf_reg = torch.mean(confidence)  # Encourage confident predictions
                loss = pred_loss + 0.1 * conf_reg
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions, confidence = self.model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions, confidence = self.model(X_tensor)
            
        return predictions.cpu().numpy(), confidence.cpu().numpy()
    
    def export_to_onnx(self, output_path: str, sequence_length: int = 50):
        """Export model to ONNX format for MQL5 integration"""
        if self.model is None:
            raise ValueError("Model must be trained before export")
            
        self.model.eval()
        
        # Create dummy input
        input_size = len(self.feature_columns) if self.feature_columns else 50
        dummy_input = torch.randn(1, sequence_length, input_size).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['prediction', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'prediction': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        # Save scaler
        scaler_path = output_path.replace('.onnx', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Model exported to {output_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def save_model(self, model_path: str):
        """Save the complete model"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'device': str(self.device)
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, input_size: int):
        """Load a saved model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model_type = checkpoint['model_type']
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        
        self.model = self.create_model(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")

class FeatureEngineer:
    """Feature engineering for HFT data"""
    
    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for HFT"""
        df = data.copy()
        
        # Price features
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['spread'] / df['mid_price']
        
        # Returns and momentum
        for period in [1, 5, 10, 20, 50]:
            df[f'return_{period}'] = df['mid_price'].pct_change(period)
            df[f'momentum_{period}'] = df['mid_price'] / df['mid_price'].shift(period) - 1
        
        # Volatility features
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['return_1'].rolling(period).std()
            df[f'realized_vol_{period}'] = np.sqrt(
                (df['return_1'] ** 2).rolling(period).sum()
            )
        
        # Volume features
        if 'volume' in df.columns:
            for period in [5, 10, 20]:
                df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Technical indicators
        df['rsi_14'] = FeatureEngineer._calculate_rsi(df['mid_price'], 14)
        df['bb_upper'], df['bb_lower'] = FeatureEngineer._calculate_bollinger_bands(
            df['mid_price'], 20, 2
        )
        df['bb_position'] = (df['mid_price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Time-based features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        # Microstructure features
        df['bid_ask_imbalance'] = (df['bid'] - df['ask']) / (df['bid'] + df['ask'])
        df['price_impact'] = df['mid_price'].diff() / df['spread']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'mid_price_lag_{lag}'] = df['mid_price'].shift(lag)
            df[f'spread_lag_{lag}'] = df['spread'].shift(lag)
        
        # Target variable (next period return)
        df['price_change'] = df['mid_price'].shift(-1) / df['mid_price'] - 1
        
        # Clean data
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

def main():
    """Example usage of the HFT predictor"""
    
    # Generate sample data (replace with real market data)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10000, freq='1min')
    
    # Simulate realistic market data
    price = 100.0
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Random walk with mean reversion
        price += np.random.normal(0, 0.001) - 0.0001 * (price - 100)
        prices.append(price)
        volumes.append(np.random.exponential(1000))
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': dates,
        'bid': np.array(prices) - np.random.uniform(0.0001, 0.001, len(prices)),
        'ask': np.array(prices) + np.random.uniform(0.0001, 0.001, len(prices)),
        'volume': volumes
    })
    data.set_index('timestamp', inplace=True)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    data_with_features = feature_engineer.create_features(data)
    
    # Train model
    trainer = HFTModelTrainer(model_type='lstm')
    X_train, X_test, y_train, y_test = trainer.prepare_data(data_with_features)
    
    # Train
    history = trainer.train(X_train, y_train, X_test, y_test, epochs=50)
    
    # Export to ONNX
    os.makedirs('../../models', exist_ok=True)
    trainer.export_to_onnx('../../models/hft_model.onnx')
    
    # Save complete model
    trainer.save_model('../../models/hft_model.pth')
    
    logger.info("Training completed and model exported!")

if __name__ == "__main__":
    main()