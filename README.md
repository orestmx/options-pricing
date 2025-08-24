# Options Pricing Calculator

A comprehensive Python application for option pricing using both analytical (Black-Scholes) and numerical (Monte Carlo) methods. Features interactive visualizations and sensitivity analysis through a modern Streamlit web interface.

**Live Demo:** [https://options-pricing-app.streamlit.app/](https://options-pricing-app.streamlit.app/)

## Features

### Black-Scholes Pricing
- Analytical option pricing using the Black-Scholes-Merton model
- Interactive sensitivity heatmaps for volatility and strike price analysis
- Real-time price calculations for both call and put options
- Put-call parity verification
- Moneyness analysis and time value decomposition

### Monte Carlo Simulation
- Geometric Brownian Motion path generation
- Configurable number of simulation paths and time steps
- Support for European and Digital option payoffs
- Antithetic variates for variance reduction
- Convergence analysis and visualization
- Path visualization and terminal price distribution analysis

### Interactive Visualizations
- Real-time sensitivity heatmaps
- Monte Carlo path visualization
- Convergence analysis plots
- Terminal price distribution histograms
- Side-by-side comparison of pricing methods

## Project Structure

```
options-pricing/
├── src/
│   ├── black_scholes/
│   │   ├── __init__.py
│   │   └── bs_pricer.py          # Black-Scholes pricing implementation
│   ├── monte_carlo/
│   │   ├── __init__.py
│   │   ├── path_generation.py    # GBM path simulation
│   │   └── payoff_func.py        # Option payoff functions
│   ├── helper_func/
│   │   ├── __init__.py
│   │   └── helper_func.py        # Utility functions
│   └── __init__.py
├── app.py                        # Streamlit web application
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/options-pricing.git
cd options-pricing
```

### Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run the Web Application
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Package Programmatically

```python
from src import bs_pricer, MCParams, gbm_path_generation, mc_price, european_payoff

# Black-Scholes pricing
call_price = bs_pricer(S=100, K=110, r=0.05, T=1.0, sigma=0.2, option_type='call')
put_price = bs_pricer(S=100, K=110, r=0.05, T=1.0, sigma=0.2, option_type='put')

# Monte Carlo simulation
cfg = MCParams(S0=100, sigma=0.2, r=0.05, T=1.0, M=100000, steps=252)
paths = gbm_path_generation(cfg)
payoff_func = european_payoff(K=110, option_type='call')
mc_price_result = mc_price(paths, payoff_func, r=0.05, T=1.0)
```

## Core Components

### Black-Scholes Module (`src.black_scholes`)
- `bs_pricer(S, K, r, T, sigma, option_type)`: Analytical option pricing

### Monte Carlo Module (`src.monte_carlo`)
- `MCParams`: Configuration class for Monte Carlo parameters
- `gbm_path_generation(cfg)`: Generate GBM price paths
- `mc_price(paths, payoff_func, r, T)`: Calculate option price from paths
- `european_payoff(K, option_type)`: European option payoff function
- `digital_payoff(K, option_type)`: Digital option payoff function

### Helper Functions (`src.helper_func`)
- `year_fraction(start_date, end_date)`: Calculate time fractions

## Application Interface

### Black-Scholes Pricing Page
- Input market parameters (spot price, strike, volatility, etc.)
- Real-time price calculations
- Interactive heatmaps showing price sensitivity to volatility and strike price
- Put-call parity verification
- Option moneyness and time value analysis

### Monte Carlo Pricing Page
- Comprehensive Monte Carlo simulation setup
- Path visualization with customizable number of displayed paths
- Convergence analysis showing Monte Carlo convergence to Black-Scholes price
- Terminal price distribution analysis
- Performance comparison between analytical and numerical methods

## Key Features

- **Real-time Calculations**: Instant price updates as parameters change
- **Interactive Visualizations**: Plotly-powered charts and heatmaps
- **Educational Focus**: Clear explanations and comparisons between methods
- **Professional Interface**: Modern, responsive design with custom styling
- **Variance Reduction**: Optional antithetic variates for improved Monte Carlo accuracy
- **Comprehensive Analysis**: Detailed statistics and convergence metrics

## Dependencies

- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black-Scholes-Merton model for analytical option pricing
- Monte Carlo methods for numerical option pricing
- Streamlit for the interactive web interface
- Plotly for advanced data visualizations

## Educational Purpose

This project is designed for educational and research purposes, providing a comprehensive platform for understanding option pricing theory and computational finance methods. It demonstrates the practical implementation of both analytical and numerical approaches to derivative pricing.

---

**Note**: This application is for educational purposes only and should not be used for actual trading decisions without proper validation and risk management procedures.
