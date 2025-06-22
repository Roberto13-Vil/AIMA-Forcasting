# Residuals diagnostics Report

## Overview
This report analyzes the residuals of models of dataset from NOAA for station **{{ station_id }}** from **{{ start_date }}** to **{{ end_date }}**.

---

## Models candidate and Justification

{{ models_candidates }}

{{ justification }}

---

## Residuals close to zero (mean = 0)

- Statistic: **{{ levene_stat }}**
- p-value: **{{ levene_pval }}**

{% if levene_pval < 0.05 %}
⚠️ Variances are significantly different.
{% else %}
✅ No significant difference in variances.
{% endif %}

---

## ADF Test (Stationarity)

- ADF Statistic: **{{ adf_stat }}**
- p-value: **{{ adf_pval }}**
- Critical values:  
{{ adf_crit_vals }}

{% if adf_pval < 0.05 %}
✅ The series is stationary.
{% else %}
⚠️ The series is non-stationary.
{% endif %}

---

## Time Series

![Time Serie](../Outputs/Figures/time_serie.png)

---

## Time Series Decomposition

![STL Decomposition](../Outputs/Figures/stl_decomposition.png)

---

## Conclusion

Based on the analysis, we observe {{ conclusion }}.