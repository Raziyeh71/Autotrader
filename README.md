# AutoTrader

An intelligent trading system that combines machine learning models, market analysis, and technical indicators to assist in making informed trading decisions.

## Features

### AI-Powered Analysis
- Multiple model comparison (LSTM, GRU, Galformer)
- Market trend prediction
- Performance metrics tracking
- LLM-based market analysis

### Technical Analysis
- RSI (Relative Strength Index)
- Moving Averages (SMA20, SMA50)
- Price action analysis
- Volume analysis

### Data Management
- Automated market data fetching
- CSV-based data storage
- Historical data analysis
- Real-time updates

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Raziyeh71/Autotrader.git
cd AutoTrader
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Run the trading system:
```bash
python autotrader.py
```

The system will:
- Fetch market data for selected symbols
- Calculate technical indicators
- Compare model performances
- Provide AI-powered market analysis
- Store results in CSV format

## Model Comparison

The system compares different models:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Galformer (Custom transformer architecture)

Performance metrics include:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Data Storage

Market data is stored in CSV format:
- Individual files per symbol
- Updated on each run
- Includes technical indicators
- Human-readable format

## CI/CD Pipeline

The project includes GitHub Actions for:
- Automated testing
- Code quality checks
- Security scanning
- Coverage reporting

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
