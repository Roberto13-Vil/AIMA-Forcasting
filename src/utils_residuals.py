import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from scipy.stats import ttest_1samp, anderson
import math
from scipy.stats import t

theme = 'seaborn-v0_8-dark-palette'
fig_facecolor = "#333333"
ax_facecolor = "#444444"

def plot_comparative(
    serie_original: pd.Series,
    serie: pd.Series,
    path: str,
    title: str,
    ylabel: str,
    xlabel: str = "Date",
    annotate: bool = True,
    show: bool = True
):
    """
    Plot and save a comparison between the original and interpolated/regularized time series.

    Parameters:
        serie_original (pd.Series): The original (possibly sparse) series.
        serie (pd.Series): The interpolated or cleaned version.
        path (str): File path to save the plot.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        annotate (bool): Whether to add an annotation arrow. Default is True.
        show (bool): Whether to display the plot. Default is True.
    """
    if not isinstance(serie.index, pd.DatetimeIndex) or not isinstance(serie_original.index, pd.DatetimeIndex):
        raise ValueError("Both series must have a DatetimeIndex.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context(theme):
        fig, ax = plt.subplots(figsize=(14, 5), facecolor=fig_facecolor)
        ax.set_facecolor(ax_facecolor)

        ax.plot(serie.index, serie, marker='o', linestyle='-', color="red", 
                label='Regularized & Interpolated', markersize=5)

        ax.plot(serie_original.index, serie_original, marker='o', linestyle='None', 
                color="purple", alpha=0.6, label='Original (dispersed)', markersize=7)

        ax.set_title(title, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.set_xlabel(xlabel, color='white')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(colors='white')

        if annotate:
            ax.annotate(
                'Missing months interpolated', 
                xy=(serie.index[10], serie.iloc[10]), 
                xytext=(serie.index[5], serie.max() + 5),
                arrowprops=dict(facecolor='white', arrowstyle='->'),
                color='white', fontsize=10
            )

        plt.savefig(path, facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close()

def fit_sarima_models(
    serie: pd.Series,
    models: dict[str, tuple[tuple[int, int, int], tuple[int, int, int, int]]],
    verbose: bool = True
) -> dict[str, any]:
    """
    Fits multiple SARIMA models on the given time series.

    Parameters:
        serie (pd.Series): Time series data.
        models (dict): Dictionary where keys are model names and values are tuples of (order, seasonal_order).
        verbose (bool): If True, prints the model being fitted.

    Returns:
        dict: Dictionary of model names and their fitted results.
    """
    results = {}

    for name, (order, seasonal_order) in models.items():
        if verbose:
            print(f'Fitting {name} ...')
        model = SARIMAX(serie, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        results[name] = fitted_model

    return results

def residual_mean_test(
    residuals: dict[str, pd.Series],
    verbose: bool = True,
    return_df: bool = False
) -> pd.DataFrame | None:
    """
    Performs a mean test (t-test) on residuals to check if the mean is significantly different from zero.

    Parameters:
        residuals (dict): Dictionary of model name → residuals (pd.Series).
        verbose (bool): Whether to print the results.
        return_df (bool): Whether to return the results as a DataFrame.

    Returns:
        pd.DataFrame | None: Summary table if return_df is True.
    """
    results = []

    for name, resid in residuals.items():
        mean = resid.mean()
        std = resid.std()
        stat, pval = ttest_1samp(resid, popmean=0)
        status = "✅ Mean ≈ 0 (good residuals)" if pval >= 0.05 else "❌ Mean ≠ 0 (bad residuals)"

        if verbose:
            print(f"{name}: t-stat = {stat:.4f}, p-value = {pval:.4f} → {status}")

        results.append({
            "Model": name,
            "Mean": mean,
            "Std": std,
            "t-stat": stat,
            "p-value": pval,
            "Status": status
        })

    if return_df:
        return pd.DataFrame(results)
    

def analyze_residual_mean(
    residuals: dict[str, pd.Series],
    save_path_plot: str,
    show_plot: bool = True,
    return_df: bool = True
) -> pd.DataFrame | None:
    """
    Analyze mean of residuals by performing a t-test and plotting them.

    Parameters:
        residuals (dict): Model name → residuals.
        save_path_plot (str): Path to save residual plots.
        show_plot (bool): Whether to display the plot.
        return_df (bool): Whether to return a DataFrame with stats.

    Returns:
        pd.DataFrame | None: Table with mean, std, t-stat, p-value, and interpretation.
    """
    results = []
    for name, resid in residuals.items():
        mean = resid.mean()
        std = resid.std()
        t_stat, pval = ttest_1samp(resid, popmean=0)
        status = "✅ Mean ≈ 0 (good residuals)" if pval >= 0.05 else "❌ Mean ≠ 0 (bad residuals)"
        results.append({
            "Model": name,
            "Mean": mean,
            "Std": std,
            "t-stat": t_stat,
            "p-value": pval,
            "Status": status
        })

    df_results = pd.DataFrame(results)

    n = len(residuals)
    cols = 3
    rows = math.ceil(n / cols)
    os.makedirs(os.path.dirname(save_path_plot), exist_ok=True)

    with plt.style.context(theme):
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), facecolor=fig_facecolor)
        axs = axs.flatten()

        for i, (name, resid) in enumerate(residuals.items()):
            ax = axs[i]
            ax.plot(resid, color='cyan', lw=1)
            ax.axhline(0, color='white', ls='--', lw=1)
            ax.set_title(f"{name} Residuals", color='white')
            ax.set_facecolor(ax_facecolor)
            ax.tick_params(colors='white')
            ax.set_xlabel("Time", color='white')
            ax.set_ylabel("Residuals", color='white')

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.tight_layout()
        plt.savefig(save_path_plot, facecolor=fig.get_facecolor())
        if show_plot:
            plt.show()
        else:
            plt.close()

    if return_df:
        return df_results

def check_stationarity_and_invertibility(
    results: dict[str, any],
    save_path_plot: str,
    show_plot: bool = True,
    return_df: bool = True
) -> pd.DataFrame | None:
    """
    Check stationarity and invertibility by analyzing the AR and MA roots of each SARIMA model.
    Also plots the roots on the complex plane with the unit circle.

    Parameters:
        results (dict): Dictionary of model name → fitted SARIMAX result object.
        save_path_plot (str): Where to save the plot.
        show_plot (bool): Whether to display the plot.
        return_df (bool): Whether to return a DataFrame with the results.

    Returns:
        pd.DataFrame | None: Table with stationarity and invertibility status.
    """
    summary = []

    for name, result in results.items():
        ar_roots = result.arroots
        ma_roots = result.maroots

        is_stationary = all(abs(root) > 1 for root in ar_roots)
        is_invertible = all(abs(root) > 1 for root in ma_roots)

        summary.append({
            "Model": name,
            "Stationary": is_stationary,
            "Invertible": is_invertible
        })

    df_summary = pd.DataFrame(summary)

    models = list(results.items())
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols

    os.makedirs(os.path.dirname(save_path_plot), exist_ok=True)

    with plt.style.context(theme):
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), facecolor=fig_facecolor)
        axs = axs.flatten()

        for i, (name, result) in enumerate(models):
            ax = axs[i]
            ax.set_facecolor(ax_facecolor)

            circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--')
            ax.add_artist(circle)

            ax.plot(result.arroots.real, result.arroots.imag, 'ro', label='AR Roots')
            ax.plot(result.maroots.real, result.maroots.imag, 'bo', label='MA Roots')

            ax.axhline(0, color='gray', linestyle='--')
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_title(name, fontsize=10, color='white')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, color='gray')
            ax.legend(loc='upper right', fontsize=8, facecolor="#555555")

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle("Roots of AR and MA Polynomials for Each Model", fontsize=16, color='white')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        plt.savefig(save_path_plot, facecolor=fig.get_facecolor())
        if show_plot:
            plt.show()
        else:
            plt.close()

    if return_df:
        return df_summary

def white_test(
    results: dict[str, any],
    verbose: bool = True,
    return_df: bool = True
) -> pd.DataFrame | None:
    """
    Applies White's test for heteroscedasticity on the residuals of SARIMA models.

    Parameters:
        results (dict): Dictionary of model name → fitted SARIMAX result object.
        verbose (bool): If True, prints the test results. Default is True.
        return_df (bool): If True, returns a DataFrame with results.

    Returns:
        pd.DataFrame | None: Table with test statistics and interpretation.
    """
    summary = []

    if verbose:
        print("White Test for Heteroscedasticity\n")

    for name, result in results.items():
        resid = result.resid
        exog = pd.DataFrame({
            'fitted': result.fittedvalues,
            'fitted_sq': result.fittedvalues**2
        })
        exog = sm.add_constant(exog)

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(resid, exog)
        status = "❌ Heteroscedasticity" if lm_pvalue < 0.05 else "✅ Homoscedasticity"

        if verbose:
            print(f"Model: {name}")
            print(f"  LM Statistic: {lm_stat:.4f}, LM p-value: {lm_pvalue:.4f}")
            print(f"  F Statistic:  {f_stat:.4f}, F p-value:  {f_pvalue:.4f}")
            print(f"  → {status}")
            print()

        summary.append({
            "Model": name,
            "LM Statistic": lm_stat,
            "LM p-value": lm_pvalue,
            "F Statistic": f_stat,
            "F p-value": f_pvalue,
            "Status": status
        })

    if return_df:
        return pd.DataFrame(summary)


def normality_residuals(
    residuals: dict[str, pd.Series],
    path: str,
    show_plot: bool = True,
    verbose: bool=True,
    return_df: bool = True
) -> pd.DataFrame | None:
    """
    Applies Anderson-Darling test to check residuals normality and plots histogram + Q-Q plots.

    Parameters:
        residuals (dict): Dictionary of model name → residuals.
        path (str): Path to save the output plot.
        show_plot (bool): Whether to display the plot.
        return_df (bool): Whether to return a summary DataFrame.

    Returns:
        pd.DataFrame | None: Summary of Anderson-Darling test results.
    """
    if verbose:
        print("Anderson-Darling Test for Normality\n")
    summary = []

    for name, resid in residuals.items():
        ad_result = anderson(resid, dist="norm")
        ad_stat = ad_result.statistic
        crit_val_5 = ad_result.critical_values[2]
        signif_lvl_5 = ad_result.significance_level[2]

        passed = ad_stat < crit_val_5
        status = "✅ Normality" if passed else "❌ Non-normality"
        if verbose:
            print(f"Model: {name}")
            print(f"  Anderson-Darling Statistic: {ad_stat:.3f}")
            print(f"  → {status} at {signif_lvl_5}% level")
            print()

        summary.append({
            "Model": name,
            "AD Statistic": ad_stat,
            "Critical Value (5%)": crit_val_5,
            "Status": status
        })

    n = len(residuals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), facecolor=fig_facecolor)
    if n == 1:
        axes = np.array([axes])

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context(theme):
        for i, (name, resid) in enumerate(residuals.items()):
            sns.histplot(resid, kde=False, stat="density", ax=axes[i, 0], bins=30,
                         color='cyan', edgecolor='white')

            mu, std = stats.norm.fit(resid)
            x = np.linspace(resid.min(), resid.max(), 100)
            p = stats.norm.pdf(x, mu, std)
            axes[i, 0].plot(x, p, linewidth=2, color='red')
            axes[i, 0].set_facecolor(ax_facecolor)
            axes[i, 0].set_title(f"{name} - Histogram", color="white")
            axes[i, 0].set_xlabel("Residuals", color="white")
            axes[i, 0].set_ylabel("Density", color="white")
            axes[i, 0].tick_params(colors='white')

            # Q-Q plot
            (x_vals, y_vals), (slope, intercept, r) = stats.probplot(resid, dist="norm")
            axes[i, 1].scatter(x_vals, y_vals, color='lime', s=10, label="Sample")
            axes[i, 1].plot(x_vals, intercept + slope * x_vals, color='red', lw=2, label="Theoretical")
            axes[i, 1].set_facecolor(ax_facecolor)
            axes[i, 1].set_title(f"{name} - Q-Q Plot", color="white")
            axes[i, 1].set_xlabel("Theoretical", color="white")
            axes[i, 1].set_ylabel("Sample", color="white")
            axes[i, 1].tick_params(colors='white')
            axes[i, 1].legend(facecolor=ax_facecolor, edgecolor="white", labelcolor="white")

        plt.suptitle("Residuals Normality Analysis", fontsize=16, color='white')
        plt.tight_layout()
        plt.savefig(path, facecolor=fig.get_facecolor())
        if show_plot:
            plt.show()
        else:
            plt.close()

    if return_df:
        return pd.DataFrame(summary)
    
def autocorrelation_residuals(
    residuals: dict[str, pd.Series],
    path: str,
    show_plot: bool = True,
    verbose: bool = True,
    return_df: bool = True
) -> pd.DataFrame | None:
    """
    Perform Ljung-Box test and plot ACF & PACF of residuals.

    Parameters:
        residuals (dict): Dictionary of model name → residuals.
        path (str): Where to save the plot.
        show_plot (bool): Whether to show the plot.
        verbose (bool): Whether to print test results.
        return_df (bool): Whether to return summary DataFrame.

    Returns:
        pd.DataFrame | None: Test results summary.
    """
    summary = []

    if verbose:
        print("Ljung-Box Test for Autocorrelation\n")

    for name, resid in residuals.items():
        lb_result = acorr_ljungbox(resid, lags=40, return_df=True)
        pval = lb_result["lb_pvalue"].iloc[-1]

        status = "✅ No autocorrelation" if pval >= 0.05 else "❌ Autocorrelation"
        if verbose:
            print(f"{name} - Ljung-Box Test (lag=40)")
            print(f"  → {status}, p-value = {pval:.4f}\n")

        summary.append({
            "Model": name,
            "Ljung-Box p-value": pval,
            "Status": status
        })

    # Plot ACF & PACF
    n = len(residuals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), facecolor=fig_facecolor)

    if n == 1:
        axes = [axes]  # si solo hay un modelo, mantener compatibilidad

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with plt.style.context('seaborn-v0_8-dark-palette'):
        fig.suptitle("ACF & PACF of Residuals", color="white")
        for i, (name, resid) in enumerate(residuals.items()):
            plot_acf(resid, ax=axes[i][0], lags=40, color="cyan")
            axes[i][0].set_title(f"{name} ACF", color="white")
            axes[i][0].set_facecolor(ax_facecolor)
            axes[i][0].tick_params(colors="white")
            axes[i][0].set_xlabel("Lags", color="white")
            axes[i][0].set_ylabel("Autocorrelation", color="white")

            plot_pacf(resid, ax=axes[i][1], lags=40, color="cyan")
            axes[i][1].set_title(f"{name} PACF", color="white")
            axes[i][1].set_facecolor(ax_facecolor)
            axes[i][1].tick_params(colors="white")
            axes[i][1].set_xlabel("Lags", color="white")
            axes[i][1].set_ylabel("Autocorrelation", color="white")

        plt.tight_layout()
        plt.savefig(path, facecolor=fig.get_facecolor())
        if show_plot:
            plt.show()
        else:
            plt.close()

    if return_df:
        return pd.DataFrame(summary)
    
def detect_outliers_with_plot(
    residuals: dict[str, pd.Series],
    save_path: str,
    alpha: float = 0.05,
    method: str = "both",  # "grubbs", "iqr", or "both"
    return_df: bool = True,
    verbose: bool = True,
    show_plot: bool = True
) -> pd.DataFrame | None:
    """
    Detects outliers in SARIMA model residuals using Grubbs and/or IQR methods, and plots boxplots.

    Parameters:
        residuals (dict): Dictionary of {model_name: residual_series}
        save_path (str): Path to save the boxplot image.
        alpha (float): Significance level for Grubbs test.
        method (str): "grubbs", "iqr", or "both".
        return_df (bool): Whether to return a summary dataframe.
        verbose (bool): Whether to print results.
        show_plot (bool): Whether to show the plot.

    Returns:
        pd.DataFrame | None: Summary table of outlier detection.
    """
    summary = []

    def grubbs_test(x, alpha):
        x = np.asarray(x)
        n = len(x)
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)
        G = np.max(np.abs(x - mean_x)) / std_x
        idx = np.argmax(np.abs(x - mean_x))

        t_dist = t.ppf(1 - alpha / (2 * n), df=n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
        return G > G_crit, idx, G, G_crit

    def iqr_outliers(x):
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers_idx = np.where((x < lower) | (x > upper))[0]
        return outliers_idx

    # Outlier detection loop
    for name, resid in residuals.items():
        entry = {"Model": name}

        if method in ["grubbs", "both"]:
            has_outlier, idx, G, G_crit = grubbs_test(resid, alpha)
            entry["Grubbs"] = "❌ Outlier" if has_outlier else "✅ None"
            entry["Grubbs_Stat"] = round(G, 3)
            entry["Grubbs_Crit"] = round(G_crit, 3)
        else:
            entry["Grubbs"] = "-"
            entry["Grubbs_Stat"] = "-"
            entry["Grubbs_Crit"] = "-"

        if method in ["iqr", "both"]:
            outliers_iqr = iqr_outliers(resid)
            entry["IQR_Outliers"] = len(outliers_iqr)
        else:
            entry["IQR_Outliers"] = "-"

        summary.append(entry)

        if verbose:
            print(f"Model: {name}")
            if method in ["grubbs", "both"]:
                print(f"  Grubbs: G = {G:.3f}, G_crit = {G_crit:.3f} → {entry['Grubbs']}")
            if method in ["iqr", "both"]:
                print(f"  IQR: {len(outliers_iqr)} outliers detected")
            print()

    # Plotting boxplots
    with plt.style.context(theme):
        n = len(residuals)
        fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), facecolor=fig_facecolor)
        if n == 1:
            axes = [axes]  # ensure iterable

        for i, (name, resid) in enumerate(residuals.items()):
            sns.boxplot(
                x=resid,
                ax=axes[i],
                boxprops=dict(facecolor="cyan", edgecolor="white"),
                whiskerprops=dict(color="white"),
                capprops=dict(color="white"),
                medianprops=dict(color="black"),
                flierprops=dict(markerfacecolor='red', markeredgecolor='red')
            )
            axes[i].set_facecolor(ax_facecolor)
            axes[i].set_title(f"{name} - Residuals Boxplot", color="white")
            axes[i].set_xlabel("Residuals", color="white")
            axes[i].tick_params(colors='white')
            axes[i].grid(True, axis="x", color="gray")

        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor())
        if show_plot:
            plt.show()
        else:
            plt.close()

    if return_df:
        return pd.DataFrame(summary)


def generate_residual_diagnostics_report(
    df_mean: pd.DataFrame,
    df_normality: pd.DataFrame,
    df_ljung: pd.DataFrame,
    df_white: pd.DataFrame,
    df_roots: pd.DataFrame,
    df_outliers: pd.DataFrame,
    image_paths: dict,
    output_path: str = "../Outputs/Report/residuals_diagnostics_report.md"
):
    """
    Generate a Markdown report for residual diagnostics.

    Parameters:
        df_mean, df_normality, df_ljung, df_white, df_roots, df_outliers: DataFrames with diagnostic results.
        image_paths (dict): Dict of section → image path.
        output_path (str): Path to save the Markdown report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def df_to_md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    with open(output_path, "w") as f:
        f.write("# Residual Diagnostics Report\n\n")

        # Section: Residual Mean
        f.write("## Residual Mean Test (t-test)\n")
        f.write(df_to_md(df_mean))
        f.write(f"\n\n![Residuals Time Series]({image_paths['residual_plot']})\n\n")

        # Section: Normality
        f.write("## Normality (Anderson-Darling Test)\n")
        f.write(df_to_md(df_normality))
        f.write(f"\n\n![Normality Plots]({image_paths['normality']})\n\n")

        # Section: Autocorrelation
        f.write("## Autocorrelation (Ljung-Box Test)\n")
        f.write(df_to_md(df_ljung))
        f.write(f"\n\n![ACF & PACF]({image_paths['acf_pacf']})\n\n")

        # Section: Heteroscedasticity
        f.write("## Heteroscedasticity (White Test)\n")
        f.write(df_to_md(df_white))
        f.write("\n")

        # Section: Stationarity & Invertibility
        f.write("## Stationarity and Invertibility (Roots of AR/MA Polynomials)\n")
        f.write(df_to_md(df_roots))
        f.write(f"\n\n![Roots Plot]({image_paths['roots']})\n\n")

        # Section: Outliers
        f.write("## Outlier Detection (Grubbs & IQR)\n")
        f.write(df_to_md(df_outliers))
        f.write(f"\n\n![Boxplot Outliers]({image_paths['boxplot_outliers']})\n\n")

        f.write("---\n")
        f.write("*Generated with Python & Statsmodels. Analyst: Roberto Vilchis.*\n")
