import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import levene
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

theme = 'seaborn-v0_8-dark-palette'
fig_facecolor = "#333333"
ax_facecolor = "#444444"

def graphic_time_series(
    serie: pd.Series, 
    title: str, 
    path: str,
    ylabel: str = "Value",
    show: bool = True
):
    """
    Plot and save a time series with a dark background style.

    Parameters:
        serie (pd.Series): Time series to plot (must have datetime index).
        title (str): Title for the plot.
        path (str): File path to save the figure (e.g., '../Outputs/Figures/plot.png').
        ylabel (str): Label for the Y-axis. Default is "Value".
        show (bool): Whether to display the plot. Default is True.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a pd.DatetimeIndex.")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context(theme):
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=fig_facecolor)
        ax.set_facecolor(ax_facecolor)

        ax.plot(serie, "o--", color="purple", label=ylabel)
        ax.set_title(title, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.set_xlabel('Date', color='white')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(colors='white')

        plt.savefig(path, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close()


def levene_test(
    serie: pd.Series,
    freq: str = '6ME',
    min_group_size: int = 5
):
    """
    Performs Levene's test to assess whether the variance of a time series 
    remains constant (homoscedasticity) or changes over time (heteroscedasticity).

    Parameters:
        serie (pd.Series): Time series with DatetimeIndex.
        freq (str): Frequency to segment the data (default is '6ME' → every 6 months).
        min_group_size (int): Minimum observations per group to be included in the test.

    Returns:
        stat (float): Levene's test statistic.
        p_value (float): p-value of the test.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a pd.DatetimeIndex.")

    groups = [group.values for _, group in serie.resample(freq) if len(group) >= min_group_size]

    if len(groups) < 2:
        raise ValueError("Not enough valid groups for Levene's test.")

    stat, p_value = levene(*groups)

    print(f"Levene's test statistic: {stat:.4f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("❌ HETEROSCEDASTICITY detected: variances differ across time segments.")
    else:
        print("✅ HOMOSCEDASTICITY confirmed: variance is stable across time.")

    return stat, p_value

def acf_plot(
    serie: pd.Series,
    lags: int,
    name:str,
    path: str,
    show: bool = True
):
    """
    Plot and save an ACF of a time series.

    Parameters:
        serie (pd.Series): Time series to plot (must have datetime index).
        lags (int): Number of lags to show.
        name (str): Title for the plot.
        path (str): File path to save the figure (e.g., '../Outputs/Figures/acf_plot.png').
        show (bool): Whether to display the plot. Default is True.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a pd.DatetimeIndex.")

    if lags >= len(serie):
        raise ValueError("Number of lags must be less than the length of the series.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context(theme):
        fig, ax = plt.subplots(figsize=(14, 4), facecolor=fig_facecolor)
        ax.set_facecolor(ax_facecolor)

        plot_acf(serie, color="cyan", lags=lags, ax=ax)

        ax.set_xlabel("Lags", color='white')
        ax.set_ylabel("Autocorrelation", color='white')
        ax.set_title(name, color='white')
        ax.tick_params(colors='white')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(path, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close()

def pacf_plot(
    serie: pd.Series,
    lags: int,
    name:str,
    path: str,
    show: bool = True
):
    """
    Plot and save a PACF of a time series.

    Parameters:
        serie (pd.Series): Time series to plot (must have datetime index).
        lags (int): Number of lags to show.
        name (str): Title for the plot.
        path (str): File path to save the figure (e.g., '../Outputs/Figures/pacf_plot.png').
        show (bool): Whether to display the plot. Default is True.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a pd.DatetimeIndex.")

    if lags >= len(serie):
        raise ValueError("Number of lags must be less than the length of the series.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context(theme):
        fig, ax = plt.subplots(figsize=(14, 4), facecolor=fig_facecolor)
        ax.set_facecolor(ax_facecolor)

        plot_pacf(serie, color="cyan", lags=lags, ax=ax)

        ax.set_xlabel("Lags", color='white')
        ax.set_ylabel("Partial Autocorrelation", color='white')
        ax.set_title(name, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.savefig(path, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close()

def adf_test(
    serie: pd.Series,
    name: str = "Series"
):
    """
    Performs the Augmented Dickey-Fuller test to check if a time series is stationary.
    
    Parameters:
        serie (pd.Series): Time series to test.
        name (str): Name of the series (for display purposes).
        return_diff (bool): If True and the series is not stationary, return the differenced series.
    
    Returns:
        stat (float): Augmented Dickey-Fuller's test statistic.
        p_value (float): p-value of the test.
    """
    stat, p_value, _, _, crit_vals, _ = adfuller(serie)

    print(f"ADF Test for '{name}':")
    print(f"  Test Statistic: {stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print("  Critical Values:")
    for key, value in crit_vals.items():
        print(f"    {key}: {value:.4f}")

    if p_value < 0.05:
        print("✅ STATIONARY: The time series is stationary (reject H0).")
    else:
        print("❌ NON-STATIONARY: The time series is not stationary (fail to reject H0).")

    return stat, p_value, crit_vals
        

def stl_plot(
    serie: pd.Series,
    path: str,
    name: str,
    period: int,
    show: bool = True
):
    """
    Performs and plots STL decomposition of a time series and saves the result.

    Parameters:
        serie (pd.Series): Time series with datetime index.
        path (str): Output path to save the figure.
        name (str): Title for the plot.
        period (int): Seasonal period for STL (e.g., 12 for monthly).
        show (bool): Whether to display the plot. Default is True.
    """
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise ValueError("The series index must be a pd.DatetimeIndex.")
    
    if period <= 1 or period >= len(serie):
        raise ValueError("Invalid period value. It must be > 1 and < len(serie).")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    stl = STL(serie, period=period)
    result = stl.fit()

    with plt.style.context(theme):
        fig, axs = plt.subplots(4, 1, figsize=(14, 8), sharex=True, facecolor=fig_facecolor)

        axs[0].plot(serie, color="white")
        axs[0].set_title(name, color="white")

        axs[1].plot(result.trend, color="orange")
        axs[1].set_title("Trend", color="white")

        axs[2].plot(result.seasonal, color="green")
        axs[2].set_title("Seasonality", color="white")

        axs[3].plot(result.resid, color="red")
        axs[3].set_title("Residual", color="white")
        axs[3].set_xlabel("Date", color="white")

        for ax in axs:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(colors='white')
            ax.set_facecolor(ax_facecolor)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.tight_layout()
        plt.savefig(path, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close()