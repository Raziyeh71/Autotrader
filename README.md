# AutoTrader

An advanced automated trading system that combines deep learning models, news analysis, and technical indicators for making informed trading decisions.

## Features

### Advanced AI Models
- **Galformer**: Custom transformer architecture with gated linear attention for market prediction
- **LSTM & GRU**: Deep learning models for time series analysis
- **Few-Shot Learning**: Pattern recognition using 10-100 examples for market similarity analysis

### News Analysis
- Real-time news monitoring from Yahoo Finance
- GPT-4 powered news impact analysis
- Sentiment analysis and market impact prediction
- News-driven opportunity detection

### Technical Analysis
- Advanced technical indicators (RSI, MACD, Bollinger Bands)
- Price action analysis
- Volume analysis
- Pattern recognition

### Market Analysis
- **LangGraph**: Market relationship graph for asset correlation analysis
- **MemGPT**: Efficient memory system for market events
- Combined analysis of news, technical indicators, and price action
- Risk assessment and confidence scoring

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AutoTrader
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `venv.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Start the trading system:
```bash
python autotrader.py
```

2. The system will:
   - Monitor news for configured symbols
   - Analyze market conditions
   - Identify trading opportunities
   - Execute trades based on confidence levels

## Trading Universe

Default trading universe includes:
- Cryptocurrencies: BTC-USD, ETH-USD
- Tech Stocks: NVDA, AAPL, MSFT, AMD, GOOGL, AMZN, TSLA

## Output Format

The system provides detailed analysis output:

```
Trading Opportunities:
+----------+----------------+------------+------+-------------+------------------+
| Symbol   | Recommendation | Confidence | Risk | News Driven | Primary Reason  |
+----------+----------------+------------+------+-------------+------------------+
| NVDA     | STRONG_BUY    | 0.92       | LOW  | Yes         | Positive AI chip|
| BTC-USD  | BUY           | 0.85       | MED  | No          | Technical break |
+----------+----------------+------------+------+-------------+------------------+
```

## Components

### NewsAgent
- Fetches real-time news from Yahoo Finance
- Analyzes news impact using GPT-4
- Maintains memory of past news analysis

### MarketAnalyzerAgent
- Combines news, technical, and price analysis
- Generates trading recommendations
- Provides confidence scores and risk assessment

### DeepLearningModels
- Implements Galformer, LSTM, and GRU models
- Includes few-shot learning for pattern recognition
- Compares model performance

## Risk Management

- Position sizing based on risk percentage
- Confidence-based trade execution
- Multi-factor risk assessment
- News-driven risk evaluation

## Dependencies

- TensorFlow for deep learning models
- PyTorch for Galformer implementation
- LangChain for graph-based analysis
- OpenAI GPT-4 for news analysis
- Beautiful Soup for news scraping
- Pandas and NumPy for data manipulation
- yfinance for market data

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[MIT License](LICENSE)
