
# Quant_Algo_trading_2025


[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rashettycode/Quant_Algo_trading_2025/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rashettycode/Quant_Algo_trading_2025/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---


📌 Overview

Quant_Algo_trading_2025 is an end-to-end algorithmic trading pipeline designed to support research, model development, and backtesting of trading strategies.
The project combines data engineering, feature engineering, machine learning models, and trading simulations to evaluate strategy performance before live deployment.

🏗️ Project Structure
Quant_Algo_trading_2025/
│
├── configs/                # YAML configs for data, features, and strategies
├── docker/                 # Dockerfile for reproducible environments
├── docs/                   # Documentation, metrics, and design notes
├── scripts/                # Data pipeline + simulation scripts
├── src/quant_trader/       # Core trading framework (features, modeling, I/O, utils)
├── tests/                  # Unit tests and smoke tests
└── outputs/                # (gitignored) Models, results, plots, logs

⚙️ Installation
1. Clone the repository
git clone https://github.com/rashettycode/Quant_Algo_trading_2025.git
cd Quant_Algo_trading_2025

2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. (Optional) Build with Docker
docker build -t quant-algo-trading .

🚀 Usage
Run the full pipeline
python scripts/run_pipeline.py

Run individual steps
python scripts/download_data.py
python scripts/build_features.py
python scripts/train_models.py
python scripts/simulate.py

Example: Tune a decision tree model
python scripts/tune_dt.py

🧪 Testing

Run unit and smoke tests to validate the pipeline:

pytest -q

📊 Quick Results

Baseline and tuned strategies have been simulated on historical market data.
Key metrics (from docs/sim_eval_best_overall.csv):

Strategy	Sharpe Ratio	Max Drawdown	CAGR	Win Rate
Buy & Hold	0.82	-32%	7.1%	53%
Baseline Model	1.12	-18%	11.4%	58%
Tuned Model	1.36	-14%	14.9%	61%

Example plot (simulation equity curve):

📚 Documentation

Architecture

Data Dictionary

Strategy Design

Evaluation Rubric

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss the change.
Make sure to update tests as needed.

📜 License

This project is licensed under the MIT License
.
