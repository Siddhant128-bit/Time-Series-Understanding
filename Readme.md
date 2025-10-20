<h1 align='center'> Time Series Learning </h1>
<br>
<hr>
<h3 align='left'>Part I: The Theoretical Bedrock of Time Series Analysis </h3>
<p>
    A robust understanding of the statistical properties inherent in time-series data is the indispensable foundation upon which all reliable forecasting models are built. <b> Without this theoretical grounding, the application of even the most sophisticated machine learning algorithms becomes a high-risk exercise in pattern-matching without comprehension, often leading to models that are brittle, misleading, and perform poorly on unseen data.</b> This section establishes that essential foundation, covering the core components of time series, the critical concept of stationarity, and the diagnostic tools used to identify underlying data structures.
</p>
<hr>
<h3 align='left'>1.1 Deconstructing Time Series: Trend, Seasonality, and Stationarity</h3>
<p>
    <b>Time series is an art and science of understanding data points ordered throughout time.</b>The initial and most crucial step of forecasting project is to decomppose the observed points into its components to understand its underlying structure.
    <br>
    These components are typically categorized as: 
    <ol>
        <li> Trend </li>
        <li> Seasonality </li>
        <li> Cyclicity </li>
    </ol>
</p>
<br>
<ol>
    <li> Trend </li> <p> &nbsp;&nbsp; It is long term underlying movement of the series.It is the persistent long run increase or decrease of the data. Such as steady growth of revenue over several years </p>
    <li> Seasonality </li><p>&nbsp;&nbsp; It is predictable, repeating pattern or fluctuations that occur at regular interval. For example, retail sales often exhibit strong weekly seasonality (higher on weekends) and yearly  </p>
    <li> Cyclicity </li><p>&nbsp;&nbsp; It is pattern that repeats over time but are not fixed. Calendar based periods, Business cycles may show expansion and contraction over several years </p>
</ol>

<p> The figure below gives general idea of all major components </p>
<img src='https://av-eks-blogoptimized.s3.amazonaws.com/98012Fig1Grph56227.png'>
<br>
<p><b>The primary objective of many forecasting methodologies is to isolate, model, and extrapolate these components to predict future values. </b></p>
<hr>

<h4> The Concept of Stationarity </h4>
<p>Among the properties of a time series, stationarity is paramount, particularly for classical statistical models. <b> A Time Series is considered stationary if its statstical properties, specifically mean, variance and autocovariance structure remains constant over time </b> In simpler terms <b> A Stationary series doesn't exhibit a trend or predictable seasonality pattern </b> Its behavior is consistant no matter when it is observed.</p>

<p> Whenever we use model like <p> ARIMA (Autoregressive Integrated Moving Average)</p> It is crucial that the time series is infact stationary in nature.Because they are designed to model relationships in a stable, time-invariant environment. Attempting to apply these models to non-stationary data can result in spurious correlations and a model that fails to generalize, producing unreliable forecasts.</p>

<img src="https://www.researchgate.net/publication/348592737/figure/fig3/AS:981645804970018@1611054006754/Examples-for-stationary-and-non-stationary-time-series.png">
<hr>

<h4> Steps for Time Series Analysis </h4>
<p> Any standard Time series analysis comprises of 2 major steps, first is to identify the characteristics of the graph (identify trend, Seasonality and Cyclicity). Then Convert the time series to stationary if needed for further analysis </p>

<ul>
    <li> Step 1: Visual Inspection for Identification </li>
    <p> The first and most intuitive step is alwys to create a time series plot of the data. <b> Visual analysis serves as the primary diagnositic tool to detect obvious features like trend and seasonality. </b> Plotting aggregated data like the mean flow for each month of the year can reveal the underlying seasonal pattern unambigiously </p>
    <li> Step 2: Differencing to have Stationarity </li>
    <p> If visual inspection reveals non-stationarity, the most common technique to stabilize the series mean is differentiating. <b> The process invovlves creating new time series of the differences between consecutive observations </b> Differencing are of following types: </p>
        <ol>
            <li> First Order Difference </li>
            <p> First order difference removes the trend, leaving a series of period over period changes.</p>
            <li> Seasonal Difference</li>
            <p> It is applied where the value from the previous season is subtracted. For monthly data with a yearly seasonality, this would be a lag-12 difference </p>
            <li> Both Trend and Seasonality </li>
            <p> To obtain stationarity time series we initially do seasonal difference to remove seasonality and if trend is still persistent on it we do first order difference on it </p>
</ul>

### 📈 Differencing in Time Series Analysis

**Differencing** is a technique used to make a **non-stationary** time series **stationary** by removing trends and/or seasonality. A stationary series has statistical properties (like mean and variance) that do not change over time, which is often a requirement for many time series forecasting models (e.g., ARIMA).

---

### 1️⃣ First-Order Differencing (for Trend)

This technique is used to remove a **linear or near-linear trend** from the data.

#### Formula

The first difference at time $t$ is the value at $t$ minus the value at the previous period ($t-1$):

$$
y_{t}' = y_{t} - y_{t-1}
$$

### Example (Removing Trend)

| Time | Original ($y_t$) | First Difference ($y_t' = y_t - y_{t-1}$) |
| :--: | :-------------: | :-------------------------------------: |
| t=1  | 10              | —                                       |
| t=2  | 12              | 2                                       |
| t=3  | 14              | 2                                       |
| t=4  | 16              | 2                                       |
| t=5  | 18              | 2                                       |

**New Series:** $2, 2, 2, 2 \rightarrow$ **✅ Stationary** (trend removed, as mean and variance are constant).

---

### 2️⃣ Seasonal Differencing (for Seasonality)

This technique is used to remove a **repeating seasonal pattern** that occurs every $s$ periods.

### Formula

The seasonal difference at time $t$ is the value at $t$ minus the value from the same period in the previous cycle ($t-s$):

$$
y_{t}' = y_{t} - y_{t-s}
$$

Where $s$ is the **seasonal period** (e.g., $s=12$ for monthly data with yearly seasonality, or $s=4$ for quarterly data).

### Example (Quarterly Data, $s=4$)

| Quarter | $y_t$ | Seasonal Difference ($y_t - y_{t-4}$) |
| :-----: | :---: | :-----------------------------------: |
| Q1\_2021 | 10    | —                                     |
| Q2\_2021 | 20    | —                                     |
| Q3\_2021 | 10    | —                                     |
| Q4\_2021 | 20    | —                                     |
| Q1\_2022 | 11    | 1 (11 - 10)                           |
| Q2\_2022 | 21    | 1 (21 - 20)                           |
| Q3\_2022 | 11    | 1 (11 - 10)                           |
| Q4\_2022 | 21    | 1 (21 - 20)                           |

**New Series:** $1, 1, 1, 1 \rightarrow$ **✅ Stationary** (seasonality removed).

---

### 3️⃣ Combined Differencing (Trend + Seasonality)

When a series exhibits **both a trend and a seasonal pattern**, both types of differencing are applied. The order generally doesn't matter mathematically, but it's often preferred to apply the **seasonal difference first** to stabilize the variance associated with seasonality.

### Step 1: Seasonal Differencing (Remove Seasonality)

First, apply the seasonal difference to the original series $y_t$:
$$
y_{t}' = y_{t} - y_{t-s}
$$

### Step 2: First-Order Differencing (Remove Remaining Trend)

Next, apply the first-order difference to the *new, seasonally-differenced* series $y_{t}'$:
$$
y_{t}'' = y_{t}' - y_{t-1}'
$$

This combined approach is typical for series like **monthly sales data** where both a long-term growth (trend) and a yearly cycle (seasonality, $s=12$) are present.
<hr>