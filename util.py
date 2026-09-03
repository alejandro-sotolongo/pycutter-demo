import pandas as pd
import streamlit as st
import pickle
from skfolio import Population, measures


def get_metric_names():
    return [
        "Annualized Mean",
        "Annualized Standard Deviation",
        "Annualized Semi-Deviation",
        "Mean Absolute Deviation",
        "Annualized Mean Less Benchmark",
        "Beta",
        "Annualized Tracking Error",
        "CVaR at 95%",
        "EVaR at 95%",
        "Worst Realization",
        "CDaR at 95%",
        "MAX Drawdown",
        "Average Drawdown",
        "EDaR at 95%",
        "First Lower Partial Moment",
        "Ulcer Index",
        "Gini Mean Difference",
        "Value at Risk at 95%",
        "Drawdown at Risk at 95%",
        "Entropic Risk Measure at 95%",
        "Fourth Central Moment",
        "Fourth Lower Partial Moment",
        "Skew",
        "Kurtosis",
        "Annualized Sharpe Ratio",
        "Annualized Sortino Ratio",
        "Annualized Information Ratio",
        "Mean Absolute Deviation Ratio",
        "First Lower Partial Moment Ratio",
        "Value at Risk Ratio at 95%",
        "CVaR Ratio at 95%",
        "Entropic Risk Measure Ratio at 95%",
        "EVaR Ratio at 95%",
        "Worst Realization Ratio",
        "Drawdown at Risk Ratio at 95%",
        "CDaR Ratio at 95%",
        "Calmar Ratio",
        "Average Drawdown Ratio",
        "EDaR Ratio at 95%",
        "Ulcer Index Ratio",
        "Gini Mean Difference Ratio",
        "Effective Number of Assets",
        "Assets Number"
    ]

def adjust_metrics(port, bench):
    summary_df = Population([port, bench]).summary()
    active_ret = (port.annualized_mean - bench.annualized_mean) * 100
    tracking_error = (port.returns - bench.returns).std() * (252 ** 0.5) * 100
    information_ratio = active_ret / tracking_error if tracking_error != 0 else float('inf')
    ret_df = pd.DataFrame({"Portfolio": port.returns, "Benchmark": bench.returns})
    beta = ret_df.Portfolio.cov(ret_df.Benchmark) / ret_df.Benchmark.var()
    # format as strings with 2 decimal places
    # percent
    summary_df.loc["Annualized Mean Less Benchmark"] = [f"{active_ret:.2f}%", "-"]
    summary_df.loc["Annualized Tracking Error"] = [f"{tracking_error:.2f}%", "-"]
    # numeric
    summary_df.loc["Beta"] = [f"{beta:.2f}", "1.00"]
    summary_df.loc["Annualized Information Ratio"] = [f"{information_ratio:.2f}", "-"]
    summary_df = summary_df.loc[get_metric_names()]
    summary_df.rename(index={"Worst Realization": "Worst Day"}, inplace=True)
    summary_df.rename(index={"CVaR at 95%": "CVaR at 95% (Daily)"}, inplace=True)
    summary_df.rename(index={"EVaR at 95%": "EVaR at 95% (Daily)"}, inplace=True)
    return summary_df

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)