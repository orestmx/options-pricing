import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import your package
import src
from src import (
    MCParams,
    gbm_path_generation,
    mc_price,
    plot_paths_density,
    european_payoff,
    digital_payoff,
    year_fraction,
    bs_pricer
)

# Streamlit App Configuration
st.set_page_config(
    page_title="Option Pricing Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {
    padding: 2rem;
}

.stMetric {
    background-color: #f0f2f6;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.stSelectbox > div > div {
    background-color: white;
}

.price-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.error-card {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.success-card {
    background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# App Title and Description
st.title("🚀 Advanced Option Pricing Calculator")
st.markdown("""
**Compare Black-Scholes and Monte Carlo pricing methods with interactive visualizations**

This app uses your custom Python pricing package to calculate option prices using both analytical (Black-Scholes) 
and numerical (Monte Carlo) methods, providing detailed comparisons and path visualizations.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Market Parameters")

    # Market parameters
    S0 = st.number_input("Spot Price (S₀)", value=100.0, min_value=0.01, step=0.01, format="%.2f")
    K = st.number_input("Strike Price (K)", value=110.0, min_value=0.01, step=0.01, format="%.2f")
    r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    T = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01, max_value=10.0, step=0.01, format="%.2f")
    sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.001, max_value=2.0, step=0.001, format="%.3f")

    st.header("⚙️ Option Settings")
    option_type = st.selectbox("Option Type", ["call", "put"])
    payoff_type = st.selectbox("Payoff Type", ["European", "Digital"])

    st.header("🎲 Monte Carlo Settings")
    M = st.number_input("Number of Paths", value=100000, min_value=1000, max_value=1000000, step=1000)
    steps = st.number_input("Time Steps", value=252, min_value=10, max_value=1000, step=1)

    # Advanced MC settings
    with st.expander("Advanced MC Settings"):
        antithetic = st.checkbox("Use Antithetic Variates", value=False)
        use_seed = st.checkbox("Use Random Seed", value=False)
        seed = st.number_input("Random Seed", value=42, min_value=0) if use_seed else None

        # For consistency with risk-neutral pricing
        st.info("Note: Drift (μ) is automatically set equal to risk-free rate (r) for risk-neutral pricing")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💰 Pricing Results")

    # Calculate button
    if st.button("🔄 Calculate Prices", type="primary"):
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Update progress
            status_text.text("Setting up Monte Carlo parameters...")
            progress_bar.progress(10)

            # Setup Monte Carlo parameters
            cfg = MCParams(
                S0=S0,
                sigma=sigma,
                r=r,
                T=T,
                M=M,
                steps=steps,
                antithetic=antithetic,
                seed=seed
            )

            # Generate paths
            status_text.text("Generating Monte Carlo paths...")
            progress_bar.progress(30)

            paths = gbm_path_generation(cfg)

            # Calculate Black-Scholes price
            status_text.text("Calculating Black-Scholes price...")
            progress_bar.progress(50)

            bs_price = bs_pricer(S=S0, K=K, r=r, T=T, sigma=sigma, option_type=option_type)

            # Calculate Monte Carlo price
            status_text.text("Calculating Monte Carlo price...")
            progress_bar.progress(70)

            if payoff_type == "European":
                payoff_func = european_payoff(K=K, option_type=option_type)
            else:  # Digital
                payoff_func = digital_payoff(K=K, option_type=option_type)

            mc_price_result = mc_price(paths, payoff_func, r, T)

            # Complete
            status_text.text("Calculations complete!")
            progress_bar.progress(100)
            time.sleep(0.5)

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Store results in session state
            st.session_state['results'] = {
                'bs_price': bs_price,
                'mc_price': mc_price_result,
                'paths': paths,
                'cfg': cfg,
                'payoff_type': payoff_type,
                'option_type': option_type,
                'K': K
            }

        except Exception as e:
            st.markdown(f"""
            <div class="error-card">
                <h4>❌ Error in Calculation</h4>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']

    with col1:
        # Price comparison metrics
        col_bs, col_mc, col_diff, col_error = st.columns(4)

        with col_bs:
            st.metric(
                label="Black-Scholes Price",
                value=f"${results['bs_price']:.4f}",
                delta=None
            )

        with col_mc:
            st.metric(
                label="Monte Carlo Price",
                value=f"${results['mc_price']:.4f}",
                delta=None
            )

        diff = abs(results['mc_price'] - results['bs_price'])
        rel_error = (diff / results['bs_price']) * 100 if results['bs_price'] != 0 else 0

        with col_diff:
            st.metric(
                label="Absolute Difference",
                value=f"${diff:.4f}",
                delta=None
            )

        with col_error:
            st.metric(
                label="Relative Error (%)",
                value=f"{rel_error:.2f}%",
                delta=None
            )

        # Convergence analysis
        if results['payoff_type'] == "European":  # Only show for European options where BS is available
            st.subheader("📈 Monte Carlo Convergence Analysis")

            # Calculate running average for convergence plot
            if results['payoff_type'] == "European":
                payoff_func = european_payoff(K=results['K'], option_type=results['option_type'])
            else:
                payoff_func = digital_payoff(K=results['K'], option_type=results['option_type'])

            # Calculate payoffs for all paths
            terminal_prices = results['paths'][:, -1]
            all_payoffs = payoff_func(terminal_prices)
            discounted_payoffs = np.exp(-r * T) * all_payoffs

            # Running average
            n_points = min(1000, len(discounted_payoffs))
            indices = np.linspace(100, len(discounted_payoffs), n_points, dtype=int)
            running_avg = [np.mean(discounted_payoffs[:i]) for i in indices]

            # Create convergence plot
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=indices,
                y=running_avg,
                mode='lines',
                name='MC Running Average',
                line=dict(color='blue', width=2)
            ))
            fig_conv.add_hline(
                y=results['bs_price'],
                line_dash="dash",
                line_color="red",
                annotation_text="Black-Scholes Price"
            )
            fig_conv.update_layout(
                title="Monte Carlo Convergence to Black-Scholes Price",
                xaxis_title="Number of Paths",
                yaxis_title="Option Price",
                height=400
            )
            st.plotly_chart(fig_conv, use_container_width=True)

with col2:
    st.header("📋 Summary")

    if 'results' in st.session_state:
        results = st.session_state['results']

        # Summary information
        summary_data = {
            "Parameter": ["Spot Price", "Strike Price", "Risk-free Rate", "Time to Maturity",
                          "Volatility", "Option Type", "Payoff Type", "MC Paths", "Time Steps"],
            "Value": [f"${S0:.2f}", f"${K:.2f}", f"{r:.1%}", f"{T:.2f} years",
                      f"{sigma:.1%}", option_type.title(), results['payoff_type'],
                      f"{M:,}", f"{steps:,}"]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Moneyness indicator
        moneyness = S0 / K
        if moneyness > 1.05:
            st.success("🟢 Deep In-The-Money")
        elif moneyness > 1.0:
            st.info("🔵 In-The-Money")
        elif moneyness > 0.95:
            st.warning("🟡 At-The-Money")
        elif moneyness > 0.9:
            st.info("🔵 Out-Of-The-Money")
        else:
            st.error("🔴 Deep Out-Of-The-Money")

    else:
        st.info("👆 Click 'Calculate Prices' to see results")

# Path visualization section
if 'results' in st.session_state:
    st.header("📊 Path Visualization")

    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        # Number of paths to plot
        n_paths_plot = st.slider("Number of Paths to Plot", 1, min(500, M), 100)

    with col_viz2:
        # Plot type selection
        plot_type = st.selectbox("Visualization Type", ["Price Paths", "Final Price Distribution", "Both"])

    if st.button("🎨 Generate Visualizations"):
        paths = results['paths']
        cfg = results['cfg']

        if plot_type in ["Price Paths", "Both"]:
            # Price paths plot
            st.subheader("Simulated Price Paths")

            # Select random paths to plot
            np.random.seed(42)  # For consistent visualization
            plot_indices = np.random.choice(paths.shape[0], size=n_paths_plot, replace=False)

            time_grid = np.linspace(0, T, steps + 1)

            fig_paths = go.Figure()

            # Add individual paths
            for i in plot_indices:
                fig_paths.add_trace(go.Scatter(
                    x=time_grid,
                    y=paths[i],
                    mode='lines',
                    line=dict(width=1, color='rgba(0,100,255,0.1)'),
                    showlegend=False,
                    hovertemplate='Time: %{x:.2f}<br>Price: %{y:.2f}<extra></extra>'
                ))

            # Add strike line
            fig_paths.add_hline(
                y=K,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Strike = ${K:.2f}"
            )

            # Add initial price line
            fig_paths.add_hline(
                y=S0,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Initial = ${S0:.2f}"
            )

            fig_paths.update_layout(
                title=f"{n_paths_plot} Simulated Price Paths",
                xaxis_title="Time (years)",
                yaxis_title="Price ($)",
                height=500
            )

            st.plotly_chart(fig_paths, use_container_width=True)

        if plot_type in ["Final Price Distribution", "Both"]:
            # Final price distribution
            st.subheader("Terminal Price Distribution")

            final_prices = paths[:, -1]

            fig_dist = go.Figure()

            # Histogram
            fig_dist.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                name='Terminal Prices',
                opacity=0.7
            ))

            # Add strike line
            fig_dist.add_vline(
                x=K,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Strike = ${K:.2f}"
            )

            # Add mean line
            mean_final = np.mean(final_prices)
            fig_dist.add_vline(
                x=mean_final,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Mean = ${mean_final:.2f}"
            )

            fig_dist.update_layout(
                title="Distribution of Terminal Prices",
                xaxis_title="Terminal Price ($)",
                yaxis_title="Frequency",
                height=400
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # Distribution statistics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Mean", f"${np.mean(final_prices):.2f}")
            with col_stat2:
                st.metric("Std Dev", f"${np.std(final_prices):.2f}")
            with col_stat3:
                st.metric("Min", f"${np.min(final_prices):.2f}")
            with col_stat4:
                st.metric("Max", f"${np.max(final_prices):.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Built with Streamlit 🚀 | Powered by your custom pricing package 📦</p>
    <p><em>For educational and research purposes</em></p>
</div>
""", unsafe_allow_html=True)