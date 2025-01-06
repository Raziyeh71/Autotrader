# Standard library imports
import os
import logging
from typing import Dict, List, Set, NamedTuple, Tuple
from datetime import datetime, timedelta
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Third-party imports
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, LayerNormalization, MultiHeadAttention, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from newspaper import Article
import networkx as nx
import pickle

# Load environment variables
load_dotenv()

# OpenAI Configuration (Optional)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not found. News analysis features will be limited.")
    client = None

class GalformerBlock(tf.keras.layers.Layer):
    """Gated Linear Attention Transformer Block"""
    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = Sequential([
            Dense(d_model * 4, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def call(self, x, training=True):
        """Forward pass through the Galformer block"""
        # Multi-head attention
        attn_output = self.attention(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class MarketMemory:
    """Enhanced memory system for market events"""
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.memory = []

    def add_market_event(self, event: Dict):
        """Add market event to memory"""
        self.memory.append({
            'timestamp': datetime.now(),
            'event': event
        })
        # Keep only recent events within token limit
        while len(str(self.memory)) > self.max_tokens:
            self.memory.pop(0)

    def get_relevant_memories(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories based on query"""
        # Simple recency-based retrieval for now
        return sorted(self.memory, 
                     key=lambda x: x['timestamp'],
                     reverse=True)[:k]

    def get_relevant(self, query: str):
        """Retrieve relevant memories based on query"""
        return [memory for memory in self.memory if query in str(memory)]

class KnowledgeTriple(NamedTuple):
    subject: str
    predicate: str
    object_: str

class MarketGraph:
    """Graph representation of market relationships"""
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_market_relationship(self, source: str, target: str, relationship: str):
        """Add relationship between market entities"""
        triple = KnowledgeTriple(subject=source, predicate=relationship, object_=target)
        self.graph.add_edge(source, target, label=relationship)
            
    def get_related_assets(self, symbol: str, relationship: str = None) -> List[str]:
        """Get assets related to given symbol"""
        try:
            if relationship:
                return [node for node in self.graph.neighbors(symbol) 
                       if self.graph.get_edge_data(symbol, node)['label'] == relationship]
            return [node for node in self.graph.neighbors(symbol)]
        except Exception as e:
            logger.warning(f"Error getting related assets for {symbol}: {e}")
            return []

class FewShotLearner:
    """Few-shot learning for market patterns"""
    def __init__(self, min_shots: int = 10, max_shots: int = 100):
        self.min_shots = min_shots
        self.max_shots = max_shots
        self.patterns = []
        self.scaler = StandardScaler()
        
    def add_pattern(self, data: np.ndarray, label: str, metadata: Dict = None):
        """Add a new pattern to the few-shot learner"""
        if len(self.patterns) >= self.max_shots:
            # Remove oldest pattern
            self.patterns.pop(0)
            
        self.patterns.append({
            'data': self.scaler.fit_transform(data),
            'label': label,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        })
        
    def find_similar_patterns(self, query_data: np.ndarray, k: int = 5) -> List[Dict]:
        """Find k most similar patterns using dynamic time warping"""
        if len(self.patterns) < self.min_shots:
            logger.warning(f"Not enough patterns for few-shot learning. Have {len(self.patterns)}, need {self.min_shots}")
            return []
            
        query_normalized = self.scaler.transform(query_data)
        similarities = []
        
        for pattern in self.patterns:
            similarity = self._calculate_similarity(query_normalized, pattern['data'])
            similarities.append({
                'similarity': similarity,
                'label': pattern['label'],
                'metadata': pattern['metadata'],
                'timestamp': pattern['timestamp']
            })
            
        # Sort by similarity and return top k
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:k]
        
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns using dynamic time warping"""
        try:
            # Normalize patterns
            p1_norm = (pattern1 - np.mean(pattern1)) / np.std(pattern1)
            p2_norm = (pattern2 - np.mean(pattern2)) / np.std(pattern2)
            
            # Calculate DTW distance
            distance = np.sum(np.abs(p1_norm - p2_norm))
            
            # Convert distance to similarity score (0 to 1)
            similarity = 1 / (1 + distance)
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

class DeepLearningModels:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.models = {}
        self.few_shot_learner = FewShotLearner(min_shots=10, max_shots=100)
        
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for deep learning models"""
        try:
            required_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
            
            # Check if all required features are present
            missing_features = [f for f in required_features if f not in data.columns]
            if missing_features:
                logger.warning(f"Missing required features: {missing_features}")
                return np.array([]), np.array([])
            
            # Handle NaN values
            data = data.ffill().bfill()
            
            # Normalize price data
            scaler = StandardScaler()
            price_cols = ['Open', 'High', 'Low', 'Close']
            data[price_cols] = scaler.fit_transform(data[price_cols])
            
            # Normalize volume separately
            data['Volume'] = scaler.fit_transform(data[['Volume']])
            
            # Keep technical indicators as is since they're already normalized
            sequences = []
            targets = []
            
            for i in range(len(data) - self.sequence_length):
                seq = data.iloc[i:i+self.sequence_length][required_features].values
                target = data.iloc[i+self.sequence_length]['Close']
                sequences.append(seq)
                targets.append(target)
            
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return np.array([]), np.array([])

    def create_model(self, input_shape: Tuple[int, int], model_type: str = 'lstm') -> tf.keras.Model:
        """Create a deep learning model"""
        try:
            inputs = tf.keras.Input(shape=input_shape)
            
            if model_type == 'lstm':
                x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
                x = tf.keras.layers.Dropout(0.3)(x)
                x = tf.keras.layers.LSTM(64)(x)
                x = tf.keras.layers.Dropout(0.2)(x)
            elif model_type == 'gru':
                x = tf.keras.layers.GRU(128, return_sequences=True)(inputs)
                x = tf.keras.layers.Dropout(0.3)(x)
                x = tf.keras.layers.GRU(64)(x)
                x = tf.keras.layers.Dropout(0.2)(x)
            else:  # galformer
                x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.3)(x)
            
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            outputs = tf.keras.layers.Dense(1)(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse')
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            raise

    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the model with optimized hyperparameters"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            verbose=0
        )
        return history

    def compare_models(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Compare performance of different models"""
        results = {}
        
        # Combine data from all symbols
        combined_data = pd.concat([df['Close'] for df in market_data.values()], axis=1)
        combined_data.columns = market_data.keys()
        
        # Prepare data for training
        X, y = self.prepare_training_data(combined_data)
        
        if len(X) < 30:
            logger.warning("Insufficient data for reliable model comparison")
            return results
            
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        total_days = len(combined_data)
        lookback = X.shape[1]
        
        print("\n=== Model Performance Analysis ===")
        print(f"Total Days of Market Data: {total_days}")
        print(f"Lookback Window: {lookback} days")
        print(f"Training Sequences: {len(X_train)} ({len(X_train)} windows of {lookback} days each)")
        print(f"Validation Sequences: {len(X_val)}")
        print(f"Testing Sequences: {len(X_test)}\n")
        
        # Train and evaluate models
        model_types = ['lstm', 'gru', 'galformer']
        model_metrics = []
        
        for model_type in model_types:
            model = self.create_model((X.shape[1], X.shape[2]), model_type)
            history = self.train_model(model, X_train, y_train, X_val, y_val)
            
            y_pred = model.predict(X_test, verbose=0)
            metrics = self.calculate_model_performance(y_pred, y_test)
            
            metrics.update({
                'Train_Loss': history.history['loss'][-1],
                'Val_Loss': history.history['val_loss'][-1],
                'Epochs_Trained': len(history.history['loss']),
                'Final_LR': float(model.optimizer.learning_rate.numpy())
            })
            
            results[model_type] = metrics
            model_metrics.append([
                model_type.upper(),
                f"{metrics['MSE']:.4f}",
                f"{metrics['MAE']:.4f}",
                f"{metrics['R2']:.4f}",
                f"Val Loss: {metrics['Val_Loss']:.4f} | Train Loss: {metrics['Train_Loss']:.4f}"
            ])
        
        # Print results table
        headers = ['Model', 'MSE', 'MAE', 'R2', 'Loss Metrics']
        print(tabulate(model_metrics, headers=headers, tablefmt='grid'))
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['MSE'])
        print(f"\nRecommended Model: {best_model[0].upper()}")
        print(f"Reasoning: Best MSE score of {best_model[1]['MSE']:.4f}\n")
        
        return results

    def calculate_model_performance(self, predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            if len(actual.shape) > 1:
                actual = actual.flatten()
                
            mse = np.mean((actual - predictions) ** 2)
            mae = np.mean(np.abs(actual - predictions))
            
            # Calculate R2 score
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            ss_res = np.sum((actual - predictions) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'MSE': float(mse),
                'MAE': float(mae),
                'R2': float(r2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {str(e)}")
            return {
                'MSE': float('inf'),
                'MAE': float('inf'),
                'R2': float('-inf')
            }

    def prepare_training_data(self, market_data: pd.DataFrame, lookback: int = 20):
        """Prepare training data from market data"""
        try:
            if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                # Normalize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(market_data)
                
                # Create sequences with overlap for more training data
                X, y = [], []
                for i in range(lookback, len(scaled_data)):  # Use all data points
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i, 0])  # Predict first symbol's price
                
                return np.array(X), np.array(y)
            else:
                logger.error("Invalid market data format")
                return np.array([]), np.array([])
                
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])

class NewsAgent:
    """Agent for fetching and analyzing financial news"""
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.model = "gpt-3.5-turbo"  
            self.client = OpenAI(api_key=self.api_key)
        else:
            logger.warning("OPENAI_API_KEY not found. News analysis features will be limited.")
            self.model = None
            self.client = None
            
    def fetch_news(self, symbol: str, days: int = 1) -> List[Dict]:
        """Fetch recent news using free sources"""
        try:
            news_items = []
            
            # Fetch from Yahoo Finance
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse news articles
            articles = soup.find_all('div', {'class': 'Py(14px)'})
            for article in articles[:10]:  # Get top 10 news items
                try:
                    title_elem = article.find('h3')
                    summary_elem = article.find('p')
                    link_elem = article.find('a')
                    
                    if title_elem and link_elem:
                        news_items.append({
                            'title': title_elem.text.strip(),
                            'summary': summary_elem.text.strip() if summary_elem else '',
                            'url': 'https://finance.yahoo.com' + link_elem['href'] if link_elem['href'].startswith('/') else link_elem['href'],
                            'source': 'Yahoo Finance',
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                except Exception as e:
                    logger.error(f"Error parsing article: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
            
    def analyze_news_impact(self, news_items: List[Dict]) -> Dict:
        """Analyze the potential market impact of news"""
        if not news_items:
            return {'impact': 'NEUTRAL', 'confidence': 0.0, 'summary': "No recent news found"}
            
        if not self.client:
            return {'impact': 'NEUTRAL', 'confidence': 0.0, 'summary': "News analysis features are limited due to missing OPENAI_API_KEY"}
            
        # Prepare news context for LLM
        news_context = "\n".join([
            f"Title: {item['title']}\nSummary: {item['summary']}\nSource: {item['source']}"
            for item in news_items
        ])
        
        # Ask LLM to analyze news impact
        prompt = f"""Analyze the following financial news and determine its potential market impact:

{news_context}

Provide your analysis in the following format:
1. Overall Impact (POSITIVE/NEGATIVE/NEUTRAL)
2. Confidence (0.0 to 1.0)
3. Key Points (bullet points)
4. Reasoning
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Analyze the news and extract trading signals."},
                {"role": "user", "content": f"Analyze this financial news and provide trading signals:\n{news_context}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Parse LLM response
        try:
            lines = response.choices[0].message.content.split('\n')
            impact = lines[0].split(':')[1].strip()
            confidence = float(lines[1].split(':')[1].strip())
            key_points = [p.strip() for p in lines[3:] if p.strip().startswith('-')]
            
            return {
                'impact': impact,
                'confidence': confidence,
                'key_points': key_points,
                'raw_analysis': response.choices[0].message.content
            }
            
        except Exception as e:
            logger.error(f"Error parsing news analysis: {e}")
            return {'impact': 'NEUTRAL', 'confidence': 0.0, 'summary': "Error analyzing news"}

class MarketAnalyzerAgent:
    """Agent for analyzing market conditions using LLM"""
    def __init__(self):
        self.openai_client = OpenAI()
        self.system_prompt = """You are an expert financial analyst. Analyze the market data and provide:
        1. Key market trends
        2. Risk assessment
        3. Trading recommendations
        Be specific and concise in your analysis."""
    
    def analyze_market_data(self, market_data: Dict[str, pd.DataFrame]) -> str:
        """Analyze market data using LLM"""
        try:
            # Prepare market summary
            summary = []
            for symbol, data in market_data.items():
                if not data.empty:
                    last_price = data['Close'].iloc[-1]
                    price_change = (last_price - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                    rsi = data['RSI'].iloc[-1]
                    summary.append(f"{symbol}: ${last_price:.2f} ({price_change:+.1f}%), RSI: {rsi:.0f}")
            
            prompt = f"""Current Market Data:
            {chr(10).join(summary)}
            
            Provide a brief analysis of the market conditions and specific trading recommendations."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return "Market analysis unavailable"

class AutoTrader:
    def __init__(self, risk_percentage: float = 5):
        """Initialize AutoTrader with advanced components"""
        self.risk_percentage = risk_percentage
        self.initial_portfolio_value = 100000
        self.current_portfolio_value = self.initial_portfolio_value
        self.positions = {}
        self.market_data = {}
        self.model_results = {}
        self.data_dir = 'market_data'
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Advanced components
        self.memory = MarketMemory()
        self.market_graph = MarketGraph()
        self.models = DeepLearningModels()
        self.news_agent = NewsAgent()
        self.market_analyzer = MarketAnalyzerAgent()
        
        # Dynamic trading universe
        self.symbols = []
        self.discover_trading_universe()
        
        # Initialize market relationships
        self._initialize_market_relationships()
        
    def fetch_market_data(self, symbols: Set[str], lookback_days: int = 180):
        """Fetch market data for the given symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        market_data = {}
        
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            csv_file = os.path.join(self.data_dir, f"{symbol}.csv")
            try:
                # Always fetch new data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Calculate technical indicators
                    data = self._calculate_indicators(data)
                    
                    # Save to CSV, overwriting old data
                    data.to_csv(csv_file)
                    market_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                # Try to use existing data as fallback
                if os.path.exists(csv_file):
                    try:
                        market_data[symbol] = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    except:
                        pass
                
        return market_data
        
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            if df.empty:
                return df
                
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate moving averages
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # Fill NaN values with backward fill
            df = df.bfill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

    def discover_trading_universe(self, min_symbols: int = 3, max_retries: int = 5) -> Set[str]:
        """Discover trading universe based on ETFs and top volume stocks"""
        try:
            # Start with major ETFs
            etfs = {'SPY', 'QQQ', 'DIA'}
            valid_symbols = set()
            
            logger.info("Discovering trading universe...")
            
            # First try ETFs
            end = datetime.now()
            start = end - timedelta(days=7)  # Use last 7 days for validation
            
            for etf in etfs:
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(start=start, end=end)
                    if not hist.empty:
                        valid_symbols.add(etf)
                except Exception as e:
                    logger.warning(f"Error validating ETF {etf}: {e}")
                    continue
                
            logger.info(f"Found {len(valid_symbols)} index ETFs")
            
            # If not enough ETFs found, look for top volume stocks
            if len(valid_symbols) < min_symbols:
                logger.info("Not enough ETFs found, searching top volume stocks...")
                
                # List of top stocks by market cap/volume
                top_stocks = [
                    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
                    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'XOM', 'BAC', 'HD',
                    'MA', 'UNH', 'CVX', 'PFE', 'CSCO'
                ]
                
                volume_threshold = 1000000  # Start with 1M volume threshold
                retry_count = 0
                
                while len(valid_symbols) < min_symbols and retry_count < max_retries:
                    for stock in top_stocks:
                        if stock in valid_symbols:
                            continue
                            
                        try:
                            ticker = yf.Ticker(stock)
                            hist = ticker.history(start=start, end=end)
                            
                            if not hist.empty and hist['Volume'].mean() > volume_threshold:
                                valid_symbols.add(stock)
                                
                            if len(valid_symbols) >= min_symbols:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error validating stock {stock}: {e}")
                            continue
                            
                    if len(valid_symbols) < min_symbols:
                        volume_threshold *= 0.5  # Reduce volume threshold by half
                        retry_count += 1
                        logger.warning("Not enough valid symbols found. Will retry with lower volume threshold.")
                        
            if len(valid_symbols) < min_symbols:
                logger.warning("No valid symbols found. Using default ETFs...")
                valid_symbols = etfs
                
            logger.info(f"Final trading universe: {valid_symbols}")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Error discovering trading universe: {e}")
            return {'SPY', 'QQQ', 'DIA'}  # Return default ETFs on error

    def discover_trading_universe(self, min_symbols: int = 3, max_retries: int = 5) -> Set[str]:
        """Discover trading universe based on ETFs and top volume stocks"""
        try:
            # Start with major ETFs
            etfs = {'SPY', 'QQQ', 'DIA'}
            valid_symbols = set()
            
            logger.info("Discovering trading universe...")
            
            # First try ETFs
            end = datetime.now()
            start = end - timedelta(days=7)  # Use last 7 days for validation
            
            for etf in etfs:
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(start=start, end=end)
                    if not hist.empty:
                        valid_symbols.add(etf)
                except Exception as e:
                    logger.warning(f"Error validating ETF {etf}: {e}")
                    continue
                
            logger.info(f"Found {len(valid_symbols)} index ETFs")
            
            # If not enough ETFs found, look for top volume stocks
            if len(valid_symbols) < min_symbols:
                logger.info("Not enough ETFs found, searching top volume stocks...")
                
                # List of top stocks by market cap/volume
                top_stocks = [
                    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
                    'JPM', 'V', 'WMT', 'JNJ', 'PG', 'XOM', 'BAC', 'HD',
                    'MA', 'UNH', 'CVX', 'PFE', 'CSCO'
                ]
                
                volume_threshold = 1000000  # Start with 1M volume threshold
                retry_count = 0
                
                while len(valid_symbols) < min_symbols and retry_count < max_retries:
                    for stock in top_stocks:
                        if stock in valid_symbols:
                            continue
                            
                        try:
                            ticker = yf.Ticker(stock)
                            hist = ticker.history(start=start, end=end)
                            
                            if not hist.empty and hist['Volume'].mean() > volume_threshold:
                                valid_symbols.add(stock)
                                
                            if len(valid_symbols) >= min_symbols:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error validating stock {stock}: {e}")
                            continue
                            
                    if len(valid_symbols) < min_symbols:
                        volume_threshold *= 0.5  # Reduce volume threshold by half
                        retry_count += 1
                        logger.warning("Not enough valid symbols found. Will retry with lower volume threshold.")
                        
            if len(valid_symbols) < min_symbols:
                logger.warning("No valid symbols found. Using default ETFs...")
                valid_symbols = etfs
                
            logger.info(f"Final trading universe: {valid_symbols}")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Error discovering trading universe: {e}")
            return {'SPY', 'QQQ', 'DIA'}  # Return default ETFs on error

    def _get_index_components(self) -> Set[str]:
        """Get components from major market indices"""
        components = set()
        indices = ['^GSPC', '^NDX', '^DJI']  # S&P 500, NASDAQ, Dow Jones
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                # Get constituents if available
                try:
                    if hasattr(ticker, 'components'):
                        components.update(ticker.components)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Error fetching components for {index}: {e}")
                
        return components
    
    def _get_news_trending_assets(self) -> Set[str]:
        """Get trending assets from financial news"""
        try:
            news_data = self.news_agent.fetch_news("market", days=3)
            trending = set()
            
            for news_item in news_data:
                try:
                    analysis = self.news_agent.analyze_news_impact([news_item])
                    if 'mentioned_assets' in analysis:
                        trending.update(analysis['mentioned_assets'])
                except:
                    continue
                    
            return trending
            
        except Exception as e:
            logger.warning(f"Error getting trending assets from news: {e}")
            return set()

    def update_trading_universe(self, lookback_days: int = 90):
        """
        Update trading universe based on correlation and clustering analysis
        """
        if not self.symbols:
            logger.warning("No symbols to analyze. Running discovery first.")
            self.discover_trading_universe()
            return

        logger.info("Updating trading universe based on market analysis...")
        
        # Fetch historical data for all symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Fetch data
        data_dict = {}
        for symbol in self.symbols:
            try:
                # Always fetch new data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate technical indicators
                    hist = self._calculate_indicators(hist)
                    
                    # Save to CSV, overwriting old data
                    hist.to_csv(os.path.join(self.data_dir, f"{symbol}.csv"))
                    data_dict[symbol] = hist
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                # Try to use existing data as fallback
                if os.path.exists(os.path.join(self.data_dir, f"{symbol}.csv")):
                    try:
                        data_dict[symbol] = pd.read_csv(os.path.join(self.data_dir, f"{symbol}.csv"), index_col=0, parse_dates=True)
                    except:
                        pass
                
        if len(data_dict) < 2:
            logger.warning("Not enough valid symbols for analysis.")
            return
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(data_dict).pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance
        dist_matrix = 1 - np.abs(corr_matrix)
        condensed_dist = squareform(dist_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='ward')
        clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
        
        # Select representatives from each cluster
        selected_symbols = []
        for cluster_id in range(1, max(clusters) + 1):
            cluster_symbols = [symbol for i, symbol in enumerate(data_dict.keys()) 
                             if clusters[i] == cluster_id]
            
            # Select symbol with highest Sharpe ratio from cluster
            cluster_returns = returns_df[cluster_symbols]
            sharpe_ratios = (cluster_returns.mean() * 252) / (cluster_returns.std() * np.sqrt(252))
            selected_symbols.append(sharpe_ratios.idxmax())
        
        self.symbols = selected_symbols
        logger.info(f"Final trading universe after analysis: {self.symbols}")

    def _initialize_market_relationships(self, lookback_days: int = 90):
        """Dynamically discover and initialize market relationships"""
        logger.info("Discovering market relationships...")
        
        try:
            # Fetch market data for correlation analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)  # Last 30 days
            
            if not self.symbols:
                self.discover_trading_universe()
            
            # Fetch data for all symbols
            data_dict = {}
            for symbol in self.symbols:
                try:
                    # Always fetch new data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        # Calculate technical indicators
                        hist = self._calculate_indicators(hist)
                        
                        # Save to CSV, overwriting old data
                        hist.to_csv(os.path.join(self.data_dir, f"{symbol}.csv"))
                        data_dict[symbol] = hist
                        
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    # Try to use existing data as fallback
                    if os.path.exists(os.path.join(self.data_dir, f"{symbol}.csv")):
                        try:
                            data_dict[symbol] = pd.read_csv(os.path.join(self.data_dir, f"{symbol}.csv"), index_col=0, parse_dates=True)
                        except:
                            pass
                    
            if len(data_dict) < 2:
                logger.warning("Not enough data to establish relationships")
                return
            
            # Calculate returns and correlation matrix
            returns_df = pd.DataFrame(data_dict).pct_change().dropna()
            corr_matrix = returns_df.corr()
            
            # Calculate volatility for each asset
            volatilities = returns_df.std()
            
            # Find market sectors using clustering
            from sklearn.cluster import AgglomerativeClustering
            
            # Convert correlation to distance matrix
            distance_matrix = 1 - np.abs(corr_matrix)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.3,  # Adjust threshold based on desired cluster granularity
                metric='precomputed',
                linkage='complete'
            )
            clusters = clustering.fit_predict(distance_matrix)
            
            # Create sector mappings
            sectors = {}
            for symbol, cluster_id in zip(corr_matrix.index, clusters):
                if cluster_id not in sectors:
                    sectors[cluster_id] = []
                sectors[cluster_id].append(symbol)
            
            # Establish relationships based on:
            # 1. High correlations
            # 2. Sector relationships
            # 3. Volatility relationships
            # 4. Lead-lag relationships (price leaders)
            
            # 1. Correlation-based relationships
            correlation_threshold = 0.7
            for i in range(len(corr_matrix.index)):
                for j in range(i + 1, len(corr_matrix.index)):
                    symbol1 = corr_matrix.index[i]
                    symbol2 = corr_matrix.index[j]
                    corr = corr_matrix.iloc[i, j]
                    
                    if abs(corr) > correlation_threshold:
                        relationship = 'strong_positive_correlation' if corr > 0 else 'strong_negative_correlation'
                        self.market_graph.add_market_relationship(symbol1, symbol2, relationship)
            
            # 2. Sector relationships
            for sector_id, sector_symbols in sectors.items():
                for i in range(len(sector_symbols)):
                    for j in range(i + 1, len(sector_symbols)):
                        self.market_graph.add_market_relationship(
                            sector_symbols[i],
                            sector_symbols[j],
                            'same_sector'
                        )
            
            # 3. Volatility relationships
            vol_sorted = volatilities.sort_values(ascending=False)
            for i in range(len(vol_sorted) - 1):
                if vol_sorted[i] > 2 * vol_sorted[i + 1]:  # Significantly more volatile
                    self.market_graph.add_market_relationship(
                        vol_sorted.index[i],
                        vol_sorted.index[i + 1],
                        'higher_volatility'
                    )
            
            # 4. Lead-lag relationships using Granger causality
            from statsmodels.tsa.stattools import grangercausalitytests
            
            for i in range(len(returns_df.columns)):
                for j in range(len(returns_df.columns)):
                    if i != j:
                        try:
                            # Use the first 30 days of data for Granger causality test
                            data = pd.concat([returns_df.iloc[:, i], returns_df.iloc[:, j]], axis=1)
                            gc_test = grangercausalitytests(data, maxlag=5, verbose=False)
                            # Check if the first asset Granger-causes the second
                            min_pvalue = min(gc_test[1][0]['ssr_chi2test'][1],
                                           gc_test[1][0]['ssr_ftest'][1])
                            if min_pvalue < 0.05:  # Statistically significant
                                self.market_graph.add_market_relationship(
                                    returns_df.columns[i],
                                    returns_df.columns[j],
                                    'price_leader'
                                )
                        except:
                            continue
            
            logger.info("Market relationships initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing market relationships: {e}")
            raise
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for the given dataframe."""
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        return df

    def _get_market_metrics(self, data: pd.DataFrame) -> Dict:
        """Get market metrics from data"""
        try:
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            return {
                'current_price': float(latest['Close']),
                'previous_price': float(previous['Close']),
                'volume': float(latest['Volume']),
                'rsi': float(latest['RSI']),
                'macd': float(latest['MACD']),
                'volatility': float(latest['Volatility']),
                'volume_ratio': float(latest['Volume_Ratio'])
            }
        except Exception as e:
            logger.error(f"Error getting market metrics: {e}")
            return {}
            
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Analyze market data using advanced components"""
        signals = {}
        
        for symbol in self.symbols:
            symbol_data = data[data['Symbol'] == symbol].copy()
            if len(symbol_data) > 0:
                # Get related assets
                related_assets = self.market_graph.get_related_assets(symbol, 'sector_peer')
                related_data = data[data['Symbol'].isin(related_assets)]
                
                # Technical analysis
                last_price = symbol_data['Close'].iloc[-1]
                rsi = symbol_data['RSI'].iloc[-1]
                macd = symbol_data['MACD'].iloc[-1]
                
                # Get relevant historical patterns
                relevant_memories = self.memory.get_relevant_memories(
                    f"Price action for {symbol}",
                    k=3
                )
                
                # Generate signal based on all factors
                signal = self._generate_signal(
                    rsi, macd,
                    symbol_data, related_data,
                    relevant_memories
                )
                
                signals[symbol] = {
                    'price': last_price,
                    'signal': signal,
                    'rsi': rsi,
                    'macd': macd
                }
        
        return signals
    
    def _generate_signal(
        self,
        rsi: float,
        macd: float,
        symbol_data: pd.DataFrame,
        related_data: pd.DataFrame,
        relevant_memories: List[str]
    ) -> str:
        """Generate trading signal using multiple factors"""
        # Basic technical signal
        signal = 'HOLD'
        if rsi < 30 and macd > 0:
            signal = 'BUY'
        elif rsi > 70 and macd < 0:
            signal = 'SELL'
            
        # Consider related assets
        if related_data is not None and not related_data.empty:
            related_momentum = related_data['Close'].pct_change().mean()
            if related_momentum > 0.02 and signal != 'SELL':
                signal = 'BUY'
            elif related_momentum < -0.02 and signal != 'BUY':
                signal = 'SELL'
                
        return signal
    
    def analyze_opportunities(self) -> List[Dict]:
        """Find trading opportunities based on news and market analysis"""
        opportunities = []
        
        for symbol in self.symbols:
            try:
                # Fetch market data
                market_data = self.fetch_market_data(symbols={symbol}, lookback_days=90)  # Increased lookback to 90 days
                if market_data.empty:
                    continue
                    
                # Get news analysis
                news_items = self.news_agent.fetch_news(symbol)
                news_analysis = self.news_agent.analyze_news_impact(news_items)
                
                # Get technical signals
                technical_signals = self._get_technical_signals(market_data)
                
                # Combine analysis
                analysis = self.market_analyzer.analyze_opportunity(
                    symbol,
                    news_analysis,
                    technical_signals,
                    market_data
                )
                
                if analysis and analysis['confidence'] > 0.7:  # Only high confidence opportunities
                    opportunities.append(analysis)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort opportunities by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Log opportunities
        self._log_opportunities(opportunities)
        
        return opportunities
    
    def _get_technical_signals(self, data: pd.DataFrame) -> Dict:
        """Get technical analysis signals"""
        try:
            close_prices = data['Close']
            
            # Calculate RSI
            rsi = self.market_analyzer._calculate_rsi(close_prices)
            
            # Calculate MACD
            macd, signal = self.market_analyzer._calculate_macd(close_prices)
            
            # Get latest values
            current_rsi = float(rsi.iloc[-1])
            current_macd = float(macd.iloc[-1])
            current_signal = float(signal.iloc[-1])
            
            # Determine signal
            if current_macd > current_signal:
                signal_type = "BUY"
            elif current_macd < current_signal:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            return {
                'rsi': current_rsi,
                'macd': current_macd,
                'signal': signal_type
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical signals: {e}")
            return {
                'rsi': 50,  # Neutral RSI
                'macd': 0,  # Neutral MACD
                'signal': "HOLD"  # Default signal
            }
    
    def _log_opportunities(self, opportunities: List[Dict]):
        """Log identified opportunities"""
        if not opportunities:
            print("\nNo high-confidence opportunities found.")
            return
            
        print("\nTrading Opportunities:")
        print("-" * 100)
        headers = ['Symbol', 'Recommendation', 'Confidence', 'Risk', 'News Driven', 'Primary Reason']
        table = []
        
        for opp in opportunities:
            table.append([
                opp['symbol'],
                opp['recommendation'],
                f"{opp['confidence']:.2f}",
                opp['risk'],
                'Yes' if opp['news_driven'] else 'No',
                textwrap.shorten(opp['primary_reason'], width=50)
            ])
            
        print(tabulate(table, headers=headers, tablefmt='grid'))
        
        # Print detailed analysis for top opportunity
        top = opportunities[0]
        print(f"\nDetailed Analysis for {top['symbol']}:")
        print("-" * 50)
        print(f"Recommendation: {top['recommendation']}")
        print(f"Confidence: {top['confidence']:.2f}")
        print(f"Risk Level: {top['risk']}")
        print("\nKey Points:")
        for point in top['key_points']:
            print(f"- {point}")

    def execute_trades(self, signals: Dict):
        """Execute trades based on signals"""
        for symbol, data in signals.items():
            current_position = self.positions.get(symbol, 0)
            
            if data['signal'] == 'BUY' and current_position == 0:
                # Calculate position size based on risk
                risk_amount = self.current_portfolio_value * (self.risk_percentage / 100)
                position_size = risk_amount / data['price']
                
                self.positions[symbol] = position_size
                trade_event = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': data['price'],
                    'size': position_size
                }
                self.memory.add_market_event(trade_event)
                
            elif data['signal'] == 'SELL' and current_position > 0:
                trade_event = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': data['price'],
                    'size': current_position
                }
                self.memory.add_market_event(trade_event)
                del self.positions[symbol]
    
    def run(self):
        """Main trading loop"""
        try:
            # Find opportunities based on news and market analysis
            opportunities = self.analyze_opportunities()
            
            # Compare model performance
            logger.info("Comparing model performance...")
            market_data = self.fetch_market_data()
            model_comparison = self.models.compare_models(market_data)
            
            # Execute trades for high-confidence opportunities
            for opp in opportunities:
                if opp['confidence'] > 0.8 and opp['recommendation'] in ['STRONG_BUY', 'BUY']:
                    self._execute_trade(opp)
            
            # Log portfolio status
            self._log_portfolio_status()
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
    
    def _log_portfolio_status(self):
        """Log current portfolio status with improved formatting"""
        print("\n📊 MARKET OVERVIEW")
        print("=" * 80)
        
        if not hasattr(self, 'market_data') or not self.market_data:
            print("No market data available")
            return
            
        market_data = []
        for symbol, data in self.market_data.items():
            if not data.empty:
                try:
                    last_price = data['Close'].iloc[-1]
                    sma20 = data.get('SMA20', data['Close']).iloc[-1]  # Default to Close if SMA20 not available
                    rsi = data.get('RSI', pd.Series([50] * len(data))).iloc[-1]  # Default to 50 if RSI not available
                    
                    trend = "↗️ Bullish" if last_price > sma20 else "↘️ Bearish"
                    safety = "🟢 Safe" if 40 <= rsi <= 60 else "🔴 Risky"
                    
                    market_data.append([
                        symbol,
                        yf.Ticker(symbol).info.get('shortName', 'N/A'),
                        f"${last_price:.2f}",
                        trend,
                        safety
                    ])
                except Exception as e:
                    logger.error(f"Error processing market data for {symbol}: {str(e)}")
                    continue
        
        if market_data:
            print(tabulate(market_data, 
                headers=['Symbol', 'Name', 'Price', 'Trend', 'Safety'],
                tablefmt='grid'))
        else:
            print("No valid market data available")
        
        # Rest of the function remains the same...

def main():
    """Main execution function"""
    try:
        trader = AutoTrader()
        
        # Discover trading universe
        symbols = trader.discover_trading_universe(min_symbols=3)
        
        if symbols:
            # Get market data and store it
            trader.market_data = trader.fetch_market_data(symbols=symbols, lookback_days=180)
            
            # Compare models and store results
            trader.model_results = trader.models.compare_models(trader.market_data)
            
            # Get LLM analysis
            llm_analysis = trader.market_analyzer.analyze_market_data(trader.market_data)
            
            # Log portfolio status with improved formatting
            trader._log_portfolio_status()
            
            if llm_analysis and llm_analysis != "Market analysis unavailable":
                print("\n📈 AI MARKET ANALYSIS")
                print("=" * 80)
                print(llm_analysis)
            
        else:
            print("\n⚠️  No valid trading symbols discovered.")
            
    except Exception as e:
        error_msg = str(e)
        solution = {
            'data': 'Ensure data source is connected and providing valid data',
            'model': 'Check model parameters and training configuration',
            'index': 'Verify data indexing and date alignment',
            'attribute': 'Confirm all required class attributes are initialized'
        }
        error_type = next((k for k in solution if k in error_msg.lower()), 'unknown')
        print(f"\n⚠️  ERROR: {error_msg}")
        print(f"💡 Solution: {solution.get(error_type, 'Contact support for assistance')}")
        logger.error(f"Error in main execution: {error_msg}")

if __name__ == "__main__":
    main()