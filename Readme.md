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
<p>Among the properties of a time series, stationarity is paramount, particularly for classical statistical models.<p>