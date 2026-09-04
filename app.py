import pandas as pd
import streamlit as st
import util

from skfolio import Portfolio, RiskMeasure, PerfMeasure, RatioMeasure, Population
from skfolio.optimization import MeanRisk, ObjectiveFunction, HierarchicalRiskParity
from scikit-learn.model_selection import train_test_split


st.set_page_config(page_title="Pycutter Demo", page_icon=":material/pie_chart:", 
    layout="wide")

@st.cache_data
def get_msl():
    msl = pd.read_parquet(
        "s3://alloc-app-demo/msl.parquet",
        storage_options={
            "key": st.secrets["s3"]["aws_access_key_id"],
            "secret": st.secrets["s3"]["aws_secret_access_key"],
        })
    msl["Lookup"] = msl.AllocName + " (" + msl.Ticker + ")"
    return msl

@st.cache_data
def get_returns():
    returns = pd.read_parquet(
        "s3://alloc-app-demo/returns.parquet",
        storage_options={
            "key": st.secrets["s3"]["aws_access_key_id"],
            "secret": st.secrets["s3"]["aws_secret_access_key"],
        })
    return returns

@st.cache_data
def get_preset_portfolios():
    return pd.read_parquet(
        "s3://alloc-app-demo/preset_ports.parquet",
        storage_options={
            "key": st.secrets["s3"]["aws_access_key_id"],
            "secret": st.secrets["s3"]["aws_secret_access_key"],
        })

msl = get_msl()
returns = get_returns()

@st.cache_data
def get_port_choices():
    preset_df = get_preset_portfolios()
    return ["Create New Portfolio"] + preset_df.Portfolio.unique().tolist()

def preset_portfolios(port=True):
    if port:
        sel = "portfolio_selectbox"
        etfs = "portfolio_etfs"
    else:
        sel = "benchmark_selectbox"
        etfs = "benchmark_etfs"

    selected = st.session_state[sel]  # Read from KEY, not portfolio_name
    if selected == "Create New Portfolio":
        st.session_state[etfs] = pd.DataFrame({
            "Asset": [],
            "Ticker": [],
            "Weight": []
        })
    else: 
        preset_df = get_preset_portfolios()
        port_df = preset_df[preset_df.Portfolio == selected]\
            [["Asset", "Ticker", "Weight"]]
        st.session_state[etfs] = port_df.reset_index(drop=True)

# init session state ---- 
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Set Up"
if "portfolio_name" not in st.session_state:
    st.session_state.portfolio_name = "US 60/40"
if "benchmark_name" not in st.session_state:
    st.session_state.benchmark_name = "Global 60/40"
if "portfolio_etfs" not in st.session_state:
    st.session_state.portfolio_etfs = pd.DataFrame(
        {"Asset": [], "Ticker": [], "Weight": []}
    )
if "benchmark_etfs" not in st.session_state:
    st.session_state.benchmark_etfs = pd.DataFrame(
        {"Asset": ["iShares MSCI All World", "Vanguard Total U.S. Bond", 
            "Vanguard Total International Bond"], 
        "Ticker": ["ACWI", "BND", "BNDX"], 
        "Weight": [0.6, 0.2, 0.2]}
    )
if "selected_port_index" not in st.session_state:
    st.session_state.selected_port_index = 0
if "selected_benchmark_index" not in st.session_state:
    st.session_state.selected_benchmark_index = 2
if "portfolio_value" not in st.session_state:
    st.session_state.portfolio_value = 100.0
if "benchmark_value" not in st.session_state:
    st.session_state.benchmark_value = 100.0
if "selected_portfolio_asset_index" not in st.session_state:
    st.session_state.selected_portfolio_asset_index = 0
if "selected_benchmark_asset_index" not in st.session_state:
    st.session_state.selected_benchmark_asset_index = 0
if "population" not in st.session_state:
    st.session_state.population = None 

# dictionaries ----
RISK_MAP = {
    "Volatility": RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    "Downside Volatility": RiskMeasure.ANNUALIZED_SEMI_DEVIATION,
    "CVaR": RiskMeasure.CVAR,
    "EVaR": RiskMeasure.EVAR,
    "Worst Day": RiskMeasure.WORST_REALIZATION,
    "Worst Drawdown": RiskMeasure.MAX_DRAWDOWN,
    "CDaR": RiskMeasure.CDAR,
    "EDaR": RiskMeasure.EDAR,
}

# top tabs ----

setup, analyze, optimize, about = st.tabs(
    ["Setup", "Analyze", "Optimize", "About"])

with setup:
    with st.expander("Portfolio Setup", expanded=True):
        st.write("Select a portfolio or create one from scratch by adding assets.")
        selected_portfolio = st.selectbox("Select a Preset Portfolio (Optional)", 
            get_port_choices(), key="portfolio_selectbox", width = 350,
            index=st.session_state.selected_port_index, on_change=preset_portfolios)
        with st.container(horizontal=True, vertical_alignment="bottom"):
            with st.popover("Add Asset", icon=":material/add:", width=400):
                st.selectbox("Select Assets to Add", [""] + msl.Lookup.tolist(),
                    key="asset_selectbox", 
                    help="Select an asset to add to your portfolio.",
                    index=st.session_state.selected_portfolio_asset_index)
                weight = st.number_input("Weight (%)", min_value=0.0, 
                    max_value=100.0, value=5.0, step=1.0) / 100
                if st.button("Add Selected Asset"):
                    selected_asset = st.session_state.asset_selectbox
                    if selected_asset and selected_asset != "":
                        new_assets = msl[msl.Lookup == selected_asset]\
                            [["AllocName", "Ticker"]]
                        new_assets = new_assets.rename(
                            columns={"AllocName": "Asset"})
                        new_assets["Weight"] = weight
                        st.session_state.portfolio_etfs = pd.concat(
                            [st.session_state.portfolio_etfs, new_assets], 
                            ignore_index=True
                        )
                        st.session_state.selected_portfolio_asset_index = 0                    
                    else:
                        st.warning("Please select at least one asset to add.")

            with st.popover("Edit Asset", icon=":material/edit:", width=400):
                if not st.session_state.portfolio_etfs.empty:
                    selected_asset = st.selectbox("Select Asset to Edit", 
                        st.session_state.portfolio_etfs.Asset.tolist(),
                        key="edit_asset_selectbox")
                    new_weight = st.number_input("New Weight (%)", min_value=0.0, 
                        max_value=100.0, value=5.0, step=1.0) / 100
                    if st.button("Update Selected Asset"):
                        if selected_asset and selected_asset != "":
                            st.session_state.portfolio_etfs.loc[
                                st.session_state.portfolio_etfs.Asset == selected_asset, 
                                "Weight"] = new_weight
                            st.session_state.selected_portfolio_asset_index = 0
                        else:
                            st.warning("Please select an asset to edit.")
                    if st.button("Remove Selected Asset"):
                        if selected_asset and selected_asset != "":
                            st.session_state.portfolio_etfs = st.session_state.portfolio_etfs[
                                st.session_state.portfolio_etfs.Asset != selected_asset]
                            st.session_state.selected_portfolio_asset_index = 0
                        else:
                            st.warning("Please select an asset to remove.")
                else:
                    st.info("No assets in the portfolio to edit. Add Assets first.")

            if st.button("Clear Portfolio"):
                st.session_state.portfolio_etfs = pd.DataFrame({
                    "Asset": [],
                    "Ticker": [],
                    "Weight": []
                })
                st.session_state.selected_port_index = 0
            if st.button("Force Weights to 100%") and \
                not st.session_state.portfolio_etfs.empty:
                    total_weight = st.session_state.portfolio_etfs.Weight.sum()
                    if total_weight > 0:
                        st.session_state.portfolio_etfs["Weight"] /= total_weight
                    else:
                        st.warning("Total weight is zero. Cannot force weights to 100%.")
            st.write(f"Total Weight: {st.session_state.portfolio_etfs.Weight.sum() * 100:.2f}%")

        st.dataframe(st.session_state.portfolio_etfs,
            hide_index=True, column_config={
                "Asset": st.column_config.TextColumn("Asset Name"),
                "Weight": st.column_config.NumberColumn("Weight", 
                    format="percent"),
            })
    
    with st.expander("Benchmark Setup (Optional, default is a Global 60 / 40)", 
        expanded=False):
        st.write("Select a benchmark or create one from scratch.")
        selected_portfolio = st.selectbox("Select a Preset Benchmark (Optional)", 
            get_port_choices(), key="benchmark_selectbox", width = 350,
            index=st.session_state.selected_benchmark_index, 
            on_change=preset_portfolios, args=(False,))
        with st.container(horizontal=True, vertical_alignment="bottom"):
            with st.popover("Add Asset", icon=":material/add:", width = 150):
                st.selectbox("Select Assets to Add", [""] + msl.Lookup.tolist(),
                    key="bench_asset_selectbox", 
                    help="Select an asset to add to your portfolio.",
                    index=st.session_state.selected_portfolio_asset_index)
                weight = st.number_input("Weight (%)", min_value=0.0, 
                    max_value=100.0, value=5.0, step=1.0, key="bench_wgt") / 100
                if st.button("Add Selected Asset", key="add_bench_asset"):
                    selected_asset = st.session_state.bench_asset_selectbox
                    if selected_asset and selected_asset != "":
                        new_assets = msl[msl.Lookup == selected_asset]\
                            [["AllocName", "Ticker"]]
                        new_assets = new_assets.rename(
                            columns={"AllocName": "Asset"})
                        new_assets["Weight"] = weight
                        st.session_state.benchmark_etfs = pd.concat(
                            [st.session_state.benchmark_etfs, new_assets], 
                            ignore_index=True
                        )
                        st.session_state.selected_benchmark_asset_index = 0                    
                    else:
                        st.warning("Please select at least one asset to add.")
            if st.button("Clear Benchmark", key="clear_bench"):
                st.session_state.benchmark_etfs = pd.DataFrame({
                    "Asset": [],
                    "Ticker": [],
                    "Weight": []
                })
                st.session_state.selected_benchmark_asset_index = 0
            if st.button("Force Weights to 100%", key="force_bench_wgt") and \
                not st.session_state.benchmark_etfs.empty:
                    total_weight = st.session_state.benchmark_etfs.Weight.sum()
                    if total_weight > 0:
                        st.session_state.benchmark_etfs["Weight"] /= total_weight
                    else:
                        st.warning("Total weight is zero. Cannot force weights to 100%.")
            st.write(f"Total Weight: {st.session_state.benchmark_etfs.Weight.sum() * 100:.2f}%")

        st.dataframe(st.session_state.benchmark_etfs,
            hide_index=True, column_config={
                "Asset": st.column_config.TextColumn("Asset Name"),
                "Weight": st.column_config.NumberColumn("Weight", 
                    format="percent"),
            })

with analyze:
    if st.session_state.portfolio_etfs.empty:
        st.warning("Please set up a portfolio in the 'Setup' tab before analyzing.")
    else:
        port_ret = returns[st.session_state.portfolio_etfs.Ticker].copy()
        if st.session_state.benchmark_etfs.empty:
            st.session_state.benchmark_etfs = pd.DataFrame({
                "Asset": ["iShares MSCI All World", "Vanguard Total U.S. Bond", 
                    "Vanguard Total International Bond"],
                "Ticker": ["ACWI", "BND", "BNDX"],
                "Weight": [0.6, 0.2, 0.2]
            })
        bench_ret = returns[st.session_state.benchmark_etfs.Ticker].copy()
        port_ret.dropna(inplace=True)
        bench_ret.dropna(inplace=True)
        intersection_index = port_ret.index.intersection(bench_ret.index)
        port_ret = port_ret.loc[intersection_index]
        bench_ret = bench_ret.loc[intersection_index]

        port = Portfolio(port_ret, 
            weights=st.session_state.portfolio_etfs.Weight.values, 
            name="Portfolio",
            tag="Portfolio")
        bench = Portfolio(bench_ret, 
            weights=st.session_state.benchmark_etfs.Weight.values, 
            name="Benchmark",
            tag="Benchmark")
        population = Population([port, bench])
        st.session_state.population = population
        date_range = f"{port_ret.index.min().strftime('%Y-%m-%d')}" \
            f" to {port_ret.index.max().strftime('%Y-%m-%d')}"

        perf, charts, inter_charts, risk_wgts = st.tabs([
            "Performance", "Charts", "Interactive Charts", "Risk Weights"])

        with perf:
            st.text(date_range)
            perf_df = util.adjust_metrics(port, bench)
            perf_df["Metric"] = perf_df.index
            perf_df = perf_df[["Metric", "Portfolio", "Benchmark"]]
            st.dataframe(perf_df, hide_index=True, height=700)

        with charts:
            fig = population.plot_cumulative_returns()
            fig.update_layout(title="Cumulative Returns", xaxis_title="Date", 
                yaxis_title="Cumulative Return")
            st.plotly_chart(fig, use_container_width=True)
            fig = population.plot_drawdowns()
            fig.update_layout(title="Drawdown", xaxis_title="Date",
                yaxis_title="Drawdown")
            st.plotly_chart(fig, use_container_width=True)
            fig = population.plot_returns_distribution()
            fig.update_layout(title="Returns Distribution", xaxis_title="Return",
                yaxis_title="Density")
            st.plotly_chart(fig, use_container_width=True)

        with inter_charts:
            chart_type = st.radio("Select Chart Type", 
                ["Risk vs Return", "Rolling"], horizontal=True)
            if chart_type == "Risk vs Return":
                include_assets = st.radio(
                    "Include Assets in Chart?", ["Yes", "No"], horizontal=True
                )
                if include_assets == "Yes":
                    population = Population([port, bench])
                    for ticker in st.session_state.portfolio_etfs.Ticker:
                        asset_port = Portfolio(
                            port_ret[[ticker]], 
                            weights=[1.0], 
                            name=ticker,
                            tag="Assets"
                        )
                        population.append(asset_port)
                else:
                    population = Population([port, bench])

                risk_measure = st.selectbox("Select Risk Measure", 
                    list(RISK_MAP.keys()), width = 300)
                fig = population.plot_measures(
                    x=RISK_MAP[risk_measure],
                    y=PerfMeasure.ANNUALIZED_MEAN
                )
                fig.update_layout(title=f"Risk vs Return ({risk_measure})", 
                    xaxis_title=risk_measure, yaxis_title="Annualized Mean")
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Rolling":
                ROLL_MAP = {
                    "Volatility": RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
                    "Downside Volatility": RiskMeasure.ANNUALIZED_SEMI_DEVIATION,
                    "CVaR": RiskMeasure.CVAR,
                    "EVaR": RiskMeasure.EVAR,
                    "Sharpe Ratio": RatioMeasure.ANNUALIZED_SHARPE_RATIO,
                    "Sortino Ratio": RatioMeasure.ANNUALIZED_SORTINO_RATIO,
                    "Calmar Ratio": RatioMeasure.CALMAR_RATIO,
                    "Annualized Mean": PerfMeasure.ANNUALIZED_MEAN,
                }
                with st.container(horizontal=True, vertical_alignment="bottom"):
                    rolling_measure = st.selectbox("Select Rolling Measure", 
                        list(ROLL_MAP.keys()), width = 300)
                    window_size = st.number_input("Select Rolling Window Size (Days)", 
                        min_value=1, max_value=252, value=63, step=1, width = 300)
                fig = population.plot_rolling_measure(
                    measure=ROLL_MAP[rolling_measure],
                    window=window_size
                )
                fig.update_layout(title=f"Rolling {rolling_measure} ({window_size} Days)", 
                    xaxis_title="Date", yaxis_title=rolling_measure, 
                    showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

        with risk_wgts:
            st.write("Risk Weights Analysis")
            risk_measure = st.selectbox("Select Risk Measure", 
                list(RISK_MAP.keys()), width = 300, key="risk_wgt_selectbox")
            fig = population.plot_contribution(
                measure=RISK_MAP[risk_measure]
            )
            fig.update_layout(title=f"Risk Contributions ({risk_measure})", 
                xaxis_title="Assets", yaxis_title="Risk Contribution")
            st.plotly_chart(fig, use_container_width=True)

with optimize:
    if st.session_state.population is None:
        st.warning("Please set up a portfolio in the 'Setup' tab before optimizing.")
    else:
        population = st.session_state.population
        OPT_RISK_MAP = RISK_MAP
        # remove EVaR and EDaR
        del OPT_RISK_MAP["EVaR"]
        del OPT_RISK_MAP["EDaR"]

        opt_type = st.radio(
            "",
            ["Max Ratio", "Frontier"],
            horizontal=True, 
        )
        risk_type = st.selectbox("Select Risk Measure:", 
                list(OPT_RISK_MAP.keys()), width = 300, key="opt_risk_selectbox")
        X_train, X_test = train_test_split(
            st.session_state.population[0].X, 
            test_size=1/3, shuffle=False
        )
        if opt_type == "Max Ratio":
            model = MeanRisk(
                risk_measure=OPT_RISK_MAP[risk_type],
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                portfolio_params={"name": "Max Ratio"}
            )

            model.fit(X_train)
            pred_model = model.predict(X_test)
            opt_population = population.append(pred_model)
            fig = population.plot_composition()
            st.plotly_chart(fig, use_container_width=True)
        
        elif opt_type == "Frontier":    
            model = MeanRisk(
                risk_measure=OPT_RISK_MAP[risk_type],
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                efficient_frontier_size=20,
                portfolio_params={"name": risk_type}
            )
            model.fit(X_train)
            population_train = model.predict(X_train)
            population_test = model.predict(X_test)
            population_train.set_portfolio_params(tag="Train")
            population_test.set_portfolio_params(tag="Test")
            population_frontier = st.session_state.population \
                + population_train + population_test
            fig = population_frontier.plot_measures(
                x=OPT_RISK_MAP[risk_type],
                y=PerfMeasure.ANNUALIZED_MEAN
            )
            st.plotly_chart(fig, use_container_width=True)
        
with about:
    st.markdown("""
    This is a demo app for my personal library. It allows the user to create a 
    portfolio from 78 ETFs that represent different asset classes and run performance
    analysis and basic optimization. The analytics and optimizations are built on
    the [skfolio](https://skfolio.org/) library. skfolio is capable of much more
    than what is shown here, and I encourage you to check it out.

    This app is not intended for financial advice or investment purposes. 

    """)


