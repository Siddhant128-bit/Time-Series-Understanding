## 100 Time Series Interview Questions & Answers

| Q | A |
| :--- | :--- |
| **Q: What is a time series?** | **A:** A sequence of observations **indexed by time** (e.g., daily sales). Order matters; often autocorrelated. |
| **Q: Difference between time series and cross-sectional data?** | **A:** **Time series**: repeated measurements over time on the same entity; **cross-sectional**: many entities measured once. Time dependence is key for time series. |
| **Q: Define stationarity.** | **A:** Statistical properties (mean, variance, autocovariance) **invariant over time**. Weak (covariance) stationarity is most used. |
| **Q: Why is stationarity important?** | **A:** Many models (ARIMA, classical inference) **assume stationarity** for consistent estimation and prediction. |
| **Q: How do you make a series stationary?** | **A:** **Differencing**, detrending, variance-stabilizing transforms (log, Box-Cox), removing seasonality. |
| **Q: What is autocorrelation (ACF)?** | **A:** **Correlation of a series with its lags**. ACF plot shows correlation at different lag values. |
| **Q: What is PACF?** | **A:** **Partial autocorrelation**: correlation at lag $k$ after removing effects of intermediate lags ($1..k-1$). |
| **Q: How to choose AR order with PACF?** | **A:** PACF **cuts off** after $p$ suggests $\text{AR}(p)$ (significant PACF up to $p$, then near zero). |
| **Q: How to choose MA order with ACF?** | **A:** ACF **cuts off** after $q$ suggests $\text{MA}(q)$ (significant ACF up to $q$, then near zero). |
| **Q: What is an AR(p) model?** | **A:** Autoregressive model: $X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} + \varepsilon_t$. |
| **Q: What is an MA(q) model?** | **A:** Moving average: $X_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + ... + \theta_q \varepsilon_{t-q}$. |
| **Q: What is ARMA(p,q)?** | **A:** **Combination of AR(p) + MA(q)** for stationary series. |
| **Q: What is ARIMA(p,d,q)?** | **A:** ARMA applied to the **$d$-th differenced series**; handles nonstationary trends (**integrated**). |
| **Q: What is seasonal ARIMA (SARIMA)?** | **A:** Adds seasonal AR/MA/d terms: $\text{ARIMA}(p,d,q)(P,D,Q)_s$ where $s$ is seasonal period. |
| **Q: What are ADF and KPSS tests?** | **A:** **ADF** (Augmented Dickey-Fuller) tests null: **unit root (nonstationary)**. **KPSS** tests null: **stationarity**. Use both for robustness. |
| **Q: What is over-differencing and why avoid it?** | **A:** Applying more differencing than needed; can introduce moving average structure and **increase variance**, hurting forecasts. |
| **Q: What's the Ljung-Box test?** | **A:** Tests for **autocorrelation in residuals** (null: no autocorrelation up to lag $m$). |
| **Q: How do you check model residuals?** | **A:** ACF/PACF of residuals, Ljung-Box, QQ plot for normality, inspect heteroscedasticity, mean $\sim 0$. |
| **Q: How to choose model order automatically?** | **A:** Use **information criteria (AIC, BIC)**, or automated routines like `auto_arima` (search over $p,d,q,P,D,Q$). |
| **Q: Difference between AIC and BIC?** | **A:** Both balance fit and complexity. **BIC penalizes complexity more strongly** (longer series $\to$ bigger penalty effect). |
| **Q: Define forecast horizon.** | **A:** **How many steps ahead you predict** (e.g., 1 day, 30 days). |
| **Q: What metrics to evaluate forecasts?** | **A:** MAE, MSE, **RMSE, MAPE, sMAPE**, RMSLE, and for probabilistic forecasts: CRPS, pinball loss. |
| **Q: When is MAPE problematic?** | **A:** When true values **near zero** $\to$ infinite or huge errors; asymmetry for over/under predictions. |
| **Q: What is cross-validation for time series?** | **A:** **Time-aware CV** (**rolling/expanding windows**) that respects temporal order; no random shuffling. |
| **Q: Explain rolling (walk-forward) validation.** | **A:** Repeatedly train on a window and test on next chunk, **moving forward in time** to simulate production forecasting. |
| **Q: What is seasonality?** | **A:** **Repeating patterns at fixed intervals** (daily, weekly, yearly). |
| **Q: How to detect seasonality?** | **A:** **ACF peaks at seasonal lags**, periodograms, seasonal decomposition (**STL**), visual plots. |
| **Q: What is STL decomposition?** | **A:** **S**easonal-**T**rend decomposition using **L**oess: separates seasonal, trend, and remainder components **robustly**. |
| **Q: What is ETS?** | **A:** **E**xponential smoothing **T**rend **S**easonality state-space models: simple methods like Holt-Winters, more generalizable via state-space. |
| **Q: When use exponential smoothing vs ARIMA?** | **A:** **ETS** good for many business series with trend/seasonality and simple structure; **ARIMA** better when autocorrelation structure is complex or residuals informative. |
| **Q: What is state-space modelling?** | **A:** Models observed data as driven by **latent (hidden) states** with observation & state transition equations (**Kalman filter** for inference). |
| **Q: Explain Kalman filter in one line.** | **A:** Recursive algorithm to estimate latent state in linear Gaussian state-space models (**predict/update**). |
| **Q: What are unit roots?** | **A:** Characteristic root = 1 leads to **nonstationarity (random walk)**. Unit root tests detect them. |
| **Q: What is cointegration?** | **A:** **Nonstationary series that have a stationary linear combination** — long-run equilibrium relationship between series. |
| **Q: What model handles cointegrated series?** | **A:** **Vector Error Correction Model (VECM)**, derived from VAR with error-correction term. |
| **Q: What is VAR?** | **A:** **Vector Autoregression**: multivariate extension where each variable depends on **lags of all variables**. |
| **Q: How choose VAR lag order?** | **A:** Use AIC/BIC/HQ or likelihood ratio tests on nested models. |
| **Q: What is Granger causality?** | **A:** $X$ Granger-causes $Y$ if **past values of $X$ add predictive power for $Y$** beyond past $Y$ alone (not true causality necessarily). |
| **Q: What is impulse response in VAR?** | **A:** Shows dynamic effect of a **shock to one variable** on current/future values of all variables. |
| **Q: What is forecast error variance decomposition?** | **A:** Proportion of forecast error variance of variable due to **shocks in each variable** in the system. |
| **Q: Describe ARCH and GARCH.** | **A:** Models for **conditional heteroscedasticity (time-varying volatility)**. $\text{GARCH}(p,q)$ includes lagged squared residuals and lagged variances. |
| **Q: When to use GARCH?** | **A:** **Financial returns** with **volatility clustering** (periods of high/low variance). |
| **Q: What is regime switching / Markov switching model?** | **A:** Parameters change according to **latent state** following Markov chain (e.g., bull vs bear markets). |
| **Q: How to handle missing data in time series?** | **A:** **Interpolation** (linear, spline), forward/backward fill, **model-based imputation** (Kalman), or remove if appropriate. |
| **Q: What is seasonal differencing?** | **A:** Difference at seasonal lag: $X_t - X_{t-s}$ to remove seasonal component. |
| **Q: Explain convolutional approaches for time series.** | **A:** **Temporal Convolutional Network (TCN)** uses dilated convolutions to capture long-range dependencies with parallelism and stable gradients. |
| **Q: Explain RNNs for time series.** | **A:** **Recurrent Neural Networks** maintain **hidden state** across time to model sequences; **LSTM/GRU** fix vanishing gradients. |
| **Q: When prefer LSTM over ARIMA?** | **A:** For **complex nonlinear dependencies**, large datasets, multivariate inputs, and when feature learning helps. |
| **Q: What are sequence-to-sequence (seq2seq) models?** | **A:** **Encoder-decoder** networks mapping input sequence to output sequence — used for multi-step forecasting. |
| **Q: How do Transformers apply to time series?** | **A:** Use **self-attention** to learn dependencies across positions; handle long-range interactions efficiently (with adaptations for time). |
| **Q: What is teacher forcing?** | **A:** During training of seq2seq, using **ground-truth previous target as next input**; speeds training but can cause exposure bias. |
| **Q: What is multi-step forecasting and two common strategies?** | **A:** Predicting multiple future steps. Strategies: **direct** (one model per horizon) and **recursive** (predict 1-step repeatedly). |
| **Q: Pros/cons of recursive vs direct multi-step forecasting?** | **A:** **Recursive**: simple but **error compounds**. **Direct**: separate models reduce propagation but require more parameters/data. |
| **Q: What is hierarchical forecasting?** | **A:** Forecasting across grouped series (e.g., SKU $\to$ category $\to$ total) and **reconciling** to ensure aggregation coherence. |
| **Q: Methods for reconciling hierarchical forecasts?** | **A:** Bottom-up, top-down, middle-out, and **optimal reconciliation (MinT — minimum trace)**. |
| **Q: What is probabilistic forecasting?** | **A:** Predict **full predictive distribution** (e.g., quantiles, intervals) rather than point estimates. |
| **Q: How produce probabilistic forecasts?** | **A:** **Quantile regression**, bootstrapping residuals, Bayesian models, or models that output distributions (e.g., Gaussian Process, deep probabilistic models). |
| **Q: What is pinball loss?** | **A:** Loss for **quantile regression**: asymmetric linear loss used to estimate quantiles. |
| **Q: What is conformal prediction for time series?** | **A:** **Distribution-free** method to produce **valid prediction intervals** via nonconformity scores plus adjustment for dependency (requires careful adaptation). |
| **Q: Explain bootstrapping time series.** | **A:** **Block bootstrap**, moving block bootstrap, or stationary bootstrap **preserve dependence** by resampling contiguous blocks, not independent observations. |
| **Q: What is a periodogram?** | **A:** Estimate of **spectral density**: shows strength of **cycles (frequencies)** in the series. |
| **Q: When use Fourier vs wavelet transforms?** | **A:** **Fourier**: global frequency content (stationary). **Wavelets**: **time-localized frequency** — good for nonstationary signals. |
| **Q: What is change-point detection?** | **A:** Identifying times where **statistical properties change** (mean, variance, distribution). Methods: CUSUM, Bayesian, binary segmentation. |
| **Q: How detect anomalies in time series?** | **A:** Residual thresholding, seasonal decomposition + outlier detection, isolation forest on features, deep models (autoencoders), rolling z-scores. |
| **Q: What is the difference between forecasting and nowcasting?** | **A:** **Forecasting**: future prediction. **Nowcasting**: estimating the current/very recent state with incomplete/lagged data. |
| **Q: What are exogenous variables in time series?** | **A:** **External predictors ($X$)** that help forecast target ($Y$). Models: ARIMAX, SARIMAX, dynamic regression. |
| **Q: How include categorical features in time series models?** | **A:** Encode (**one-hot, embeddings**), include as exogenous regressors, or use feature crosses. Keep time alignment. |
| **Q: What is feature engineering for time series?** | **A:** **Lags**, **rolling statistics** (mean/std), **time features** (hour/day/week), **Fourier terms** for seasonality, holiday indicators. |
| **Q: How to handle trend and seasonality simultaneously?** | **A:** Decompose (STL), include polynomial/linear trend terms, **seasonal dummies or Fourier terms**, or use models that capture them (ETS, SARIMA). |
| **Q: What is a unit root vs deterministic trend?** | **A:** **Unit root = stochastic trend (random walk)**. **Deterministic trend** = fixed function of time (e.g., linear) — differencing vs detrending differ. |
| **Q: How to test for structural breaks?** | **A:** **Chow test** (known break point), **Bai-Perron** multiple break tests, or recursive methods/CUSUM. |
| **Q: What is backtesting and why important?** | **A:** **Simulate historical forecasting process** to evaluate models realistically (**rolling** or expanding windows) before deployment. |
| **Q: What is concept drift in time series?** | **A:** **Distribution or relationship changes over time** (covariate shift, label shift). Requires monitoring and model retraining. |
| **Q: How to detect concept drift?** | **A:** Monitor forecast errors or input distribution metrics (KS test, population stability index), or use **drift detectors (ADWIN)**. |
| **Q: What are transfer learning approaches for time series?** | **A:** Pretrain on many related series (**global models**) and fine-tune on target; use embeddings or multi-task learning. |
| **Q: What is a global (pooled) model vs local model?** | **A:** **Global model**: single model trained across many series. **Local**: separate model per series. Global helps when individual series are short. |
| **Q: Explain Prophet (Facebook) basic idea.** | **A:** **Additive model** with trend, seasonality (Fourier series), and holiday effects with robust fitting — **easy to use** for business time series. |
| **Q: What are Fourier terms and why use them?** | **A:** **Sine/cosine features** to model periodic seasonality **smoothly**, useful for regression or machine learning models. |
| **Q: How to forecast intermittent demand?** | **A:** **Croston's method**, Syntetos-Boylan approximation (SBA), or specialized probabilistic models for zero-inflated counts. |
| **Q: What is hierarchical time series reconciliation (MinT) in one line?** | **A:** Adjust base forecasts to make them **aggregate consistently**, **minimizing overall forecast error covariance** (uses covariance matrix). |
| **Q: How to evaluate probabilistic forecasts?** | **A:** **CRPS** (continuous ranked probability score), **proper scoring rules**, **calibration** (PIT), sharpness, and coverage of intervals. |
| **Q: Explain Gaussian Processes (GP) for time series.** | **A:** **Nonparametric Bayesian approach** modeling function with **covariance kernel**; yields mean & uncertainty; kernels encode smoothness/periodicity. |
| **Q: Typical GP kernels for time series?** | **A:** RBF (squared exponential), periodic, Matérn, and **sums/products** to capture trend + seasonality. |
| **Q: What is the curse of dimensionality in multivariate TS?** | **A:** **Parameter explosion** and data scarcity as number of series/variables grow; need regularization, low-rank, or dimension reduction. |
| **Q: What is Dynamic Time Warping (DTW)?** | **A:** **Distance metric** **aligning sequences by warping time axis**, useful for clustering or similarity search (not forecasting). |
| **Q: How to do time series clustering?** | **A:** Feature-based (extract stats), **shape-based (DTW)**, model-based (mixture of ARIMA/Gaussians). |
| **Q: What is spectral density?** | **A:** **How variance is distributed across frequencies**; estimated via periodogram or Welch method. |
| **Q: Explain backpropagation through time (BPTT).** | **A:** Training RNNs by **unfolding through time** and applying backpropagation across time steps; can have vanishing/exploding gradients. |
| **Q: What are attention mechanisms' benefits for TS?** | **A:** Learn **which time steps/inputs matter** for predictions, handle long-range dependencies and improve interpretability. |
| **Q: How to handle very sparse time series?** | **A:** Aggregation to coarser granularity, pooling across series (**global models**), specialized intermittent methods, or zero-inflated models. |
| **Q: What is the difference between interpolation and imputation?** | **A:** **Interpolation** fills gaps using neighboring values (deterministic). **Imputation** may use model-based or stochastic methods considering other variables. |
| **Q: What production considerations matter for time series models?** | **A:** Data pipelines, **latency, retraining frequency, monitoring** (prediction drift, data quality), scaling, and explainability for stakeholders. |
| **Q: How to monitor a deployed forecasting model?** | **A:** Track key metrics (prediction error, coverage), input data stats, alerts on drift, and periodically **re-evaluate performance via backtesting**. |
| **Q: What is explainability for time series models?** | **A:** Techniques: **SHAP**/feature importance on engineered features, **attention visualization**, partial dependence on time features, residual analysis. |
| **Q: How to tune hyperparameters for TS models?** | **A:** **Time-aware CV (rolling)**, grid/random/Bayesian search; ensure validation respects temporal order. |
| **Q: How does scaling/normalization work for TS?** | **A:** Scale **per series** (e.g., z-score) or global; be careful to fit scaler **only on training data** and invert transform on forecasts. |
| **Q: What is inverse transform sampling in forecasting?** | **A:** Generating **sample paths** by drawing random numbers from model predictive distribution; used for probabilistic scenarios. |
| **Q: How to combine forecasts (ensembling)?** | **A:** **Simple averages, weighted averages** (based on past errors), stacking/meta-learners; ensembles often improve robustness. |
| **Q: When to use manually engineered features vs end-to-end deep learning?** | **A:** If data is limited or domain knowledge strong, engineered features help. For large datasets, end-to-end may learn features automatically. |
| **Q: Give a short checklist when starting a time-series project.** | **A:** 1) Understand business goal/horizon. 2) Inspect & visualize series. 3) Check stationarity/seasonality. 4) Clean/missing handling. 5) Feature engineering/exogenous variables. 6) Baseline models (naive, ETS). 7) Model selection & CV (rolling). 8) Residual diagnostics. 9) Probabilistic evaluation & calibration. 10) Deployment & monitoring plan. |

***