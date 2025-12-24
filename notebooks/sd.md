# ARIMA Model Mastery Assessment

## How to Use This Document

This is your **self-assessment and learning guide**. Work through each section sequentially:
1. **Attempt to answer each question yourself first** (write your answer down)
2. **Then reveal the answer** to check your understanding
3. **If you got it wrong**, re-read the explanation and make notes

I've organized this from **foundational concepts → code comprehension → practical application**.

---

# PART 1: CONCEPTUAL FOUNDATION

## Section 1.1: Stationarity (The Most Critical Concept)

### Question 1.1.1
**Why does ARIMA require stationary data? What breaks if you fit ARIMA on non-stationary data?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**
ARIMA assumes that the statistical properties (mean, variance, autocorrelation) remain **constant over time**. If data is non-stationary:

1. **Parameters become meaningless** — The AR/MA coefficients are estimated assuming a stable relationship. If the mean is drifting, these relationships don't hold.

2. **Forecasts explode or collapse** — Non-stationary data can have "unit roots" where shocks have permanent effects, causing forecasts to drift unboundedly.

3. **Spurious relationships** — You might find significant correlations that don't actually exist (spurious regression).

**Analogy:** It's like trying to predict someone's height by measuring them on an escalator — the baseline keeps changing, so your measurements are useless for prediction.

</details>

---

### Question 1.1.2
**In your notebook, the ADF test returned a p-value > 0.05 for raw crude oil prices. Explain in plain English what the ADF test is actually checking.**

<details>
<summary>Click to reveal answer</summary>

**Answer:**
The ADF test is checking: **"Does this series behave like a random walk?"**

More specifically, it tests whether the series has a **unit root**. A unit root means:
- Today's value = Yesterday's value + random shock
- Shocks have **permanent** effects (they never fade away)
- The series has no tendency to return to any mean

**The hypotheses:**
- H₀ (Null): Series HAS a unit root (non-stationary, like a drunk person's walk)
- H₁ (Alternative): Series is stationary (like a dog on a leash — wanders but returns)

**p-value > 0.05** means we FAIL to reject H₀, so we conclude the series likely has a unit root and needs differencing.

</details>

---

### Question 1.1.3
**What does "differencing" actually do mathematically? Write the equation for first differencing and explain what pattern it removes.**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**First differencing equation:**
```
Y'_t = Y_t - Y_{t-1}
```

**What it does:** Instead of modeling the actual price, you model the **change** in price from one period to the next.

**What pattern it removes:** 
- **Linear trends** — If prices are steadily increasing by $1/day, differencing converts this to a constant series of 1s
- **Stochastic trends (unit roots)** — Random walk behavior is converted to white noise

**Crude oil example:**
- Original: [70, 72, 71, 74, 73] (non-stationary, drifting)
- Differenced: [+2, -1, +3, -1] (stationary, fluctuating around 0)

**Key insight:** After differencing, your model predicts *changes*, not levels. The final forecast must be "un-differenced" (cumulative sum) to get back to price levels.

</details>

---

## Section 1.2: ACF/PACF Interpretation

### Question 1.2.1
**What is the fundamental difference between ACF and PACF? Why do we need both?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**ACF (Autocorrelation Function):**
- Measures correlation between Y_t and Y_{t-k} **including all intermediate effects**
- If lag 1 is correlated, lag 2 will automatically appear correlated (through lag 1)
- Shows **total** correlation at each lag

**PACF (Partial Autocorrelation Function):**
- Measures correlation between Y_t and Y_{t-k} **after removing the effect of all lags in between**
- "Pure" correlation at each lag, controlling for shorter lags
- Shows **direct** correlation at each lag

**Why we need both:**
- **PACF cutoff** → Identifies AR order (p)
- **ACF cutoff** → Identifies MA order (q)

**Analogy:** 
- ACF is like asking "Is your grandfather's height correlated with yours?" (Yes, through your parents)
- PACF is like asking "Is your grandfather's height correlated with yours, AFTER accounting for your parents' heights?" (Much weaker, only genetic skip)

</details>

---

### Question 1.2.2
**Complete this table from memory:**

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| AR(p) | ? | ? |
| MA(q) | ? | ? |
| ARMA(p,q) | ? | ? |

<details>
<summary>Click to reveal answer</summary>

| Model | ACF Pattern | PACF Pattern |
|-------|-------------|--------------|
| **AR(p)** | Tails off (exponential decay) | **Cuts off after lag p** |
| **MA(q)** | **Cuts off after lag q** | Tails off (exponential decay) |
| **ARMA(p,q)** | Tails off | Tails off |

**Memory trick:**
- **AR** → **P**ACF cuts off (both have the letter P... sort of)
- **MA** → **A**CF cuts off (MA and ACF both start with vowels... ok, weak mnemonic)

**Better memory trick:**
- AR is about **past values** → PACF shows direct past value effects
- MA is about **past errors** → ACF captures cumulative error effects

</details>

---

### Question 1.2.3
**In your notebook, you looked at ACF/PACF of the differenced crude oil data and chose ARIMA(1,1,1). Based on the theory, what pattern would you expect to see in the plots to justify p=1 and q=1?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

For ARIMA(1,1,1), the differenced series follows ARMA(1,1). You'd expect:

**PACF pattern for p=1:**
- Significant spike at lag 1
- All subsequent lags fall within confidence bands (no significant spikes at lag 2, 3, etc.)
- This suggests the direct effect of past values only extends 1 period back

**ACF pattern for q=1:**
- Significant spike at lag 1
- Gradual decay OR cutoff after lag 1
- Since it's mixed ARMA(1,1), the ACF won't cut off sharply — it will show a spike at lag 1 then decay

**Reality check for crude oil:**
In practice, crude oil ACF/PACF plots are often ambiguous because:
1. The signal is weak (close to random walk)
2. There might be slight spikes at multiple lags
3. This is why auto_arima and AIC comparison are valuable

</details>

---

## Section 1.3: Model Components

### Question 1.3.1
**Write out the full equation for ARIMA(1,1,1) in expanded form (not using backshift notation). Define every term.**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

Let W_t = Y_t - Y_{t-1} (the differenced series)

**ARIMA(1,1,1) equation:**
```
W_t = c + φ₁·W_{t-1} + ε_t + θ₁·ε_{t-1}
```

**Or substituting back:**
```
(Y_t - Y_{t-1}) = c + φ₁·(Y_{t-1} - Y_{t-2}) + ε_t + θ₁·ε_{t-1}
```

**Term definitions:**
- **Y_t**: Current crude oil price
- **Y_{t-1}**: Yesterday's price
- **c**: Constant/drift term (often small or zero)
- **φ₁** (phi): AR(1) coefficient — how much yesterday's *change* affects today's change
- **θ₁** (theta): MA(1) coefficient — how much yesterday's *forecast error* affects today
- **ε_t**: Today's innovation/shock (unpredictable random error)
- **ε_{t-1}**: Yesterday's innovation

**Interpretation for crude oil:**
- φ₁ > 0: Price changes have momentum (positive change → more likely positive change)
- φ₁ < 0: Price changes mean-revert (positive change → likely negative change tomorrow)
- θ₁ captures how forecast errors propagate

</details>

---

### Question 1.3.2
**What does it mean for an AR coefficient to be "statistically significant" (p-value < 0.05)? What should you do if it's NOT significant?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**Statistical significance means:**
The coefficient is reliably different from zero. There's strong evidence that this term actually contributes to explaining/predicting the series.

- **Significant (p < 0.05):** The AR/MA term captures real structure in the data
- **Not significant (p > 0.05):** The coefficient might just be capturing noise; could be zero

**What to do if not significant:**

1. **Consider removing that term** — Try ARIMA(0,1,1) instead of ARIMA(1,1,1) if AR is insignificant
2. **Compare AIC** — Fit both models and see which has lower AIC
3. **Don't blindly remove** — Sometimes insignificant terms still improve forecasting slightly

**For crude oil specifically:**
It's common for both AR and MA terms to be marginally significant or even insignificant because oil prices are close to a random walk. This doesn't mean the model is useless — it means the predictable component is weak.

</details>

---

## Section 1.4: Diagnostics

### Question 1.4.1
**What is the Ljung-Box test checking? If p-value < 0.05, what does this tell you about your model?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**What it tests:**
The Ljung-Box test checks whether the residuals (errors) of your model are **white noise** — meaning they are random with no autocorrelation.

**Hypotheses:**
- H₀: Residuals are white noise (no autocorrelation)
- H₁: Residuals have significant autocorrelation

**Interpretation:**
- **p-value > 0.05:** Good! Residuals appear random. Your model captured the structure.
- **p-value < 0.05:** Bad! There's still pattern left in the errors.

**If p < 0.05, your model is incomplete because:**
1. Wrong p or q order — information at certain lags wasn't captured
2. Structural break — model was fit on heterogeneous regimes
3. Non-linear effects — ARIMA can't capture these (might need GARCH for volatility)
4. Seasonality — might need SARIMA

**Action:** Examine residual ACF to see which lags have remaining autocorrelation, then adjust model accordingly.

</details>

---

### Question 1.4.2
**Your notebook mentions "volatility clustering" and GARCH. Why might ARIMA residuals show volatility clustering even if Ljung-Box passes, and what does this mean?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**Volatility clustering definition:**
Periods of high volatility tend to cluster together, as do periods of low volatility. Big price swings follow big price swings; calm periods follow calm periods.

**Why Ljung-Box can pass but volatility clustering exists:**
- Ljung-Box tests autocorrelation in the **level** of residuals (ε_t)
- Volatility clustering is autocorrelation in the **squared** residuals (ε_t²)
- The mean of residuals can be zero with no autocorrelation, but the variance can still change over time

**What this means:**
1. Your ARIMA model correctly captures the **mean dynamics** (expected price movement)
2. But it assumes **constant variance**, which is wrong for crude oil
3. The uncertainty around your forecasts is wrong — sometimes it should be wider, sometimes narrower

**Solution:** 
- ARIMA-GARCH: Use ARIMA for mean, GARCH for variance
- This gives you both better point forecasts AND better confidence intervals

**For crude oil:**
Volatility clustering is extremely common. OPEC announcements, geopolitical events, and market stress create clustered volatility. This is why professional oil forecasters almost always use GARCH extensions.

</details>

---

# PART 2: CODE COMPREHENSION

Now let's test your understanding of the actual Python code.

## Section 2.1: Data Loading & Preparation

### Question 2.1.1
**Explain what this code does line by line:**

```python
data = yf.download('CL=F', start='2020-01-01', end='2024-01-01')

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = data[['Close']].copy()
```

<details>
<summary>Click to reveal answer</summary>

**Line 1:** `data = yf.download('CL=F', start='2020-01-01', end='2024-01-01')`
- Downloads historical data for WTI Crude Oil Futures (ticker 'CL=F') from Yahoo Finance
- Date range: Jan 1, 2020 to Jan 1, 2024
- Returns a DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
- Index is DatetimeIndex

**Lines 2-3:** `if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)`
- **Why this exists:** Recent yfinance versions return MultiIndex columns like ('Close', 'CL=F')
- `isinstance()` checks if columns are MultiIndex type
- `get_level_values(0)` extracts just the first level ('Close', 'Open', etc.)
- This flattens the column names for easier access

**Line 4:** `df = data[['Close']].copy()`
- `data[['Close']]` selects only the Close column as a DataFrame (double brackets keep it as DataFrame, not Series)
- `.copy()` creates an independent copy to avoid SettingWithCopyWarning
- We only need closing prices for ARIMA (it's a univariate model)

</details>

---

### Question 2.1.2
**What would happen if you wrote `data['Close']` instead of `data[['Close']]`? Why might this matter?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**`data['Close']`** → Returns a **pandas Series** (1D)
**`data[['Close']]`** → Returns a **pandas DataFrame** (2D with one column)

**Why it matters:**

1. **Method availability:** Some operations work differently on Series vs DataFrame
   - Series has no `.columns` attribute
   - DataFrame allows adding new columns easily: `df['new_col'] = ...`

2. **Shape consistency:** Later in your notebook, you do:
   ```python
   df['diff_1'] = df['Close'].diff()
   ```
   This works because `df` is a DataFrame. If it were a Series, you can't add a new column to it.

3. **statsmodels compatibility:** Many statsmodels functions expect specific input shapes. Some want 1D arrays (Series), others want DataFrames.

**Best practice:** Use `[['Close']]` when you want to maintain DataFrame structure and might add more columns later.

</details>

---

## Section 2.2: ADF Test Function

### Question 2.2.1
**Examine this function. What does `result[0]` and `result[1]` contain? What other information does `adfuller()` return that you're not using?**

```python
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("Conclusion: Stationary (d=0)")
    else:
        print("Conclusion: Non-Stationary (needs differencing)")
```

<details>
<summary>Click to reveal answer</summary>

**Answer:**

`adfuller()` returns a tuple with 6 elements:

| Index | Content | Description |
|-------|---------|-------------|
| `result[0]` | **ADF Statistic** | The test statistic (more negative = stronger evidence against unit root) |
| `result[1]` | **p-value** | Probability of observing this statistic if H₀ is true |
| `result[2]` | **Lags Used** | Number of lags included in the regression |
| `result[3]` | **Number of Observations** | Sample size used in the test |
| `result[4]` | **Critical Values** | Dictionary with 1%, 5%, 10% critical values |
| `result[5]` | **Information Criterion** | AIC value used to select lag length |

**What you're NOT using but could be valuable:**

```python
result = adfuller(timeseries)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"Lags Used: {result[2]}")
print(f"Observations: {result[3]}")
print(f"Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value:.4f}")
```

**Why critical values matter:**
The ADF statistic uses non-standard distribution. Critical values help you see:
- How far your test statistic is from rejection thresholds
- Whether you're marginally non-stationary or strongly non-stationary

</details>

---

## Section 2.3: Differencing

### Question 2.3.1
**What does `.diff()` do, and why does it create a NaN in the first row? What happens if you don't `.dropna()`?**

```python
df['diff_1'] = df['Close'].diff()
df_diff = df.dropna()
```

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**What `.diff()` does:**
```python
df['diff_1'][t] = df['Close'][t] - df['Close'][t-1]
```
For each row, it subtracts the previous row's value.

**Why first row is NaN:**
- Row 0 has no "previous row" to subtract
- `df['Close'][0] - df['Close'][-1]` is undefined
- Pandas correctly returns NaN rather than guessing

**What happens without `.dropna()`:**

1. **ADF test might fail:** `adfuller()` doesn't handle NaN gracefully — you'll get an error or wrong results

2. **ACF/PACF will be affected:** Missing values can distort correlation calculations

3. **ARIMA fitting will fail:** statsmodels will raise an error about missing values

**Alternative approaches:**
```python
# Option 1: Drop NaN (you used this)
df_diff = df.dropna()

# Option 2: Fill with 0 (not recommended for first difference)
df['diff_1'] = df['Close'].diff().fillna(0)

# Option 3: Use periods parameter for higher-order differencing
df['diff_12'] = df['Close'].diff(periods=12)  # Seasonal differencing
```

</details>

---

## Section 2.4: Model Fitting

### Question 2.4.1
**Explain the train/test split. Why 80/20, and what's the critical mistake many beginners make with time series splits?**

```python
train_size = int(len(df) * 0.8)
train, test = df['Close'][:train_size], df['Close'][train_size:]
```

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**What the code does:**
- Calculates 80% of total data length
- `train` = first 80% of observations (chronologically earlier)
- `test` = last 20% of observations (chronologically later)

**Why 80/20:**
- Common convention balancing training data volume vs. test set reliability
- For 4 years of data (roughly 1000 trading days), gives ~800 train, ~200 test
- 200 days is enough to assess forecast quality across different market conditions

**THE CRITICAL MISTAKE: Random splits**

```python
# WRONG for time series!
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

**Why this is catastrophic:**
1. **Data leakage:** Future data points end up in training, past in testing
2. **Your model "sees the future"** — artificially inflated accuracy
3. **Autocorrelation violation:** Time dependencies are broken

**Correct approach (what you did):**
- **Always chronological split** — train on past, test on future
- This mimics real-world forecasting where you only have historical data

**Even better: Walk-forward validation**
```python
# Your rolling forecast does this correctly!
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,1)).fit()
    forecast = model.forecast()
    history.append(actual_observation)
```

</details>

---

### Question 2.4.2
**What does `model_fit.summary()` contain? Identify the 5 most important things to check.**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

The summary output contains (in order of importance for you):

**1. Coefficient p-values (P>|z| column)**
- Tests if AR/MA terms are significantly different from zero
- **Check:** All should be < 0.05 ideally
- If not significant, consider removing that term

**2. AIC/BIC values**
- Model selection criteria (lower = better)
- **Check:** Compare across different (p,d,q) specifications
- AIC is better for forecasting; BIC for "true model" identification

**3. Coefficient values and signs**
- AR coefficient (ar.L1): Sign indicates momentum (+) vs mean-reversion (-)
- MA coefficient (ma.L1): How shocks propagate
- **Check:** Magnitude should be < 1 for stationarity/invertibility

**4. Log Likelihood**
- Higher = better fit
- Basis for AIC/BIC calculation
- **Check:** Make sure model converged (no warnings)

**5. Ljung-Box Q-statistic (if shown)**
- Tests residual autocorrelation
- **Check:** p-value > 0.05 means residuals are white noise

**Quick checklist:**
```
☐ All coefficients significant (p < 0.05)?
☐ AIC lower than alternative models?
☐ AR/MA coefficients |value| < 1?
☐ No convergence warnings?
☐ Residuals pass Ljung-Box (p > 0.05)?
```

</details>

---

## Section 2.5: Forecasting

### Question 2.5.1
**What's the difference between `model_fit.forecast()` and `model_fit.get_forecast()`? When would you use each?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**`model_fit.forecast(steps=n)`**
- Returns: Just the point forecasts (predicted values)
- Type: numpy array or pandas Series
- Use when: You only need predictions, not uncertainty

```python
predictions = model_fit.forecast(steps=10)
# Returns: array([75.2, 75.4, 75.3, ...])
```

**`model_fit.get_forecast(steps=n)`**
- Returns: A `PredictionResults` object with multiple attributes
- Includes: Point forecasts AND confidence intervals
- Use when: You need uncertainty quantification

```python
forecast_result = model_fit.get_forecast(steps=10)
point_forecast = forecast_result.predicted_mean  # Same as forecast()
confidence_int = forecast_result.conf_int()       # Lower and upper bounds
```

**Your notebook correctly uses `get_forecast()` because:**
1. You want to plot the "cone of uncertainty"
2. Confidence intervals are essential for risk assessment in trading
3. Shows model uncertainty growing over forecast horizon

**Confidence interval interpretation:**
- Default is 95% CI (alpha=0.05)
- Wider intervals = more uncertainty
- For crude oil, intervals grow quickly because predictability is low

</details>

---

### Question 2.5.2
**Explain the rolling forecast loop step by step. Why is this more realistic than multi-step-ahead forecasting?**

```python
history = train.values.flatten().tolist()
predictions = []

for t in range(len(test)):
    model_rolling = ARIMA(history, order=model_auto.order).fit()
    output = model_rolling.forecast()
    yhat = output[0]
    predictions.append(yhat)
    
    obs = test.iloc[t]
    history.append(obs)
```

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**Step-by-step breakdown:**

1. **`history = train.values.flatten().tolist()`**
   - Converts training data to a simple Python list
   - `.flatten()` ensures 1D array
   - `.tolist()` converts numpy array to list (easier to append)

2. **`for t in range(len(test)):`**
   - Iterates through each day in test set
   - t=0 is first test day, t=1 is second, etc.

3. **`model_rolling = ARIMA(history, order=model_auto.order).fit()`**
   - Creates NEW model using all available history
   - Uses same (p,d,q) order as auto_arima found
   - Model is re-estimated with updated data each iteration

4. **`output = model_rolling.forecast()`**
   - Predicts just the NEXT value (1-step ahead)
   - Not trying to predict 10 days out

5. **`predictions.append(yhat)`**
   - Stores the prediction for later comparison

6. **`obs = test.iloc[t]` + `history.append(obs)`**
   - Gets the ACTUAL observed price for that day
   - Adds it to history for next iteration's model
   - This is key: we use real data, not our predictions!

**Why this is more realistic:**

| Multi-step ahead | Rolling 1-step ahead |
|------------------|---------------------|
| Fit model once | Re-fit model each step |
| Predict all 200 days at once | Predict 1 day at a time |
| Errors compound | Errors don't accumulate |
| Assumes no new info | Incorporates actual outcomes |
| Unrealistic | Mimics real trading |

**In real trading:**
- At market close, you see actual price
- You incorporate that into your model
- You make prediction for tomorrow only
- This is exactly what rolling forecast simulates

**Error reduction:**
- Multi-step RMSE: ~$6-8 (error compounds over 200 days)
- Rolling RMSE: ~$1-3 (error is fresh each day)

</details>

---

## Section 2.6: Auto-ARIMA

### Question 2.6.1
**Explain what each parameter in `auto_arima()` does:**

```python
model_auto = pm.auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=True
)
```

<details>
<summary>Click to reveal answer</summary>

**Answer:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `train` | your data | The time series to model |
| `start_p=0` | 0 | Begin search for AR order at p=0 |
| `start_q=0` | 0 | Begin search for MA order at q=0 |
| `max_p=5` | 5 | Maximum AR order to consider |
| `max_q=5` | 5 | Maximum MA order to consider |
| `d=None` | None | **Let auto_arima determine d** using unit root tests |
| `seasonal=False` | False | Don't look for seasonal patterns (ARIMA not SARIMA) |
| `stepwise=True` | True | Use stepwise algorithm instead of exhaustive grid search |
| `suppress_warnings=True` | True | Don't print convergence warnings |
| `error_action="ignore"` | "ignore" | Skip models that fail to converge |
| `trace=True` | True | Print each model tried and its AIC |

**Key insights:**

**`d=None`:** Auto_arima will run KPSS/ADF tests internally to determine differencing order. You could also set `d=1` if you already know from your manual test.

**`stepwise=True`:** Instead of trying all 6×6×3 = 108 combinations, it uses a smart search:
1. Starts with a base model
2. Varies one parameter at a time
3. Moves to better models
4. Stops when no improvement found
Much faster (tries ~20-30 models instead of 100+)

**`seasonal=False`:** Critical for crude oil! You explicitly disable seasonal search because:
1. Crude oil seasonality is statistically weak
2. SARIMA often doesn't beat ARIMA for oil
3. Faster computation

**If you wanted SARIMA:**
```python
model_auto = pm.auto_arima(
    train,
    seasonal=True,
    m=12,  # Monthly seasonality (if monthly data)
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    D=None,  # Let it determine seasonal differencing
)
```

</details>

---

# PART 3: PRACTICAL APPLICATION

## Section 3.1: Decision Making

### Question 3.1.1
**Your auto_arima selected ARIMA(2,1,2) with AIC=5800, but your manual ARIMA(1,1,1) had AIC=5850. However, the (1,1,1) model has all significant coefficients while (2,1,2) has one insignificant coefficient. Which model do you choose and why?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**This is a judgment call. Here's the framework:**

**Arguments for ARIMA(2,1,2):**
- Lower AIC (5800 < 5850)
- AIC already penalizes complexity, so if it's lower, the extra parameters are "worth it"
- Better in-sample fit

**Arguments for ARIMA(1,1,1):**
- All coefficients significant
- More parsimonious (fewer parameters)
- Less risk of overfitting
- More interpretable
- Likely more stable out-of-sample

**My recommendation: ARIMA(1,1,1)**

**Reasoning:**
1. **AIC difference is small** (50 points, ~1% difference) — not decisive
2. **Insignificant coefficients are a red flag** — suggesting overfitting
3. **For crude oil specifically**, simpler models often forecast better because the signal is weak
4. **Out-of-sample performance matters more than in-sample AIC**

**Better approach: Out-of-sample validation**
```python
# Fit both models, compare actual test RMSE
rmse_111 = evaluate_model(train, test, order=(1,1,1))
rmse_212 = evaluate_model(train, test, order=(2,1,2))
print(f"ARIMA(1,1,1) RMSE: {rmse_111}")
print(f"ARIMA(2,1,2) RMSE: {rmse_212}")
# Choose whichever has lower RMSE on unseen data
```

</details>

---

### Question 3.1.2
**Your Ljung-Box test returned p-value = 0.03. The residual ACF shows a small but significant spike at lag 7. What are your options?**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**Diagnosis:** p-value < 0.05 means residuals are NOT white noise. There's structure you haven't captured.

**Option 1: Increase AR or MA order**
```python
# Try adding more lags
model = ARIMA(train, order=(7, 1, 1))  # AR(7) to capture lag 7
# OR
model = ARIMA(train, order=(1, 1, 7))  # MA(7)
```
**Pros:** Might capture the lag-7 effect
**Cons:** May overfit; why specifically lag 7?

**Option 2: Investigate the lag-7 pattern**
- Lag 7 in daily data = weekly effect
- Check if there's a day-of-week pattern (e.g., Monday effect)
- This might be spurious or related to weekly inventory reports

**Option 3: Accept imperfection**
- p=0.03 is marginal (close to 0.05)
- One small spike at lag 7 might be noise
- If forecasting performance is acceptable, the model might be "good enough"
- Crude oil is notoriously hard to model perfectly

**Option 4: Use GARCH for the variance**
- The autocorrelation might be in squared residuals (volatility)
- ARIMA-GARCH could resolve this

**Option 5: Try auto_arima with different settings**
```python
model_auto = pm.auto_arima(train, max_p=10, max_q=10, ...)
```

**What I'd actually do:**
1. Check if lag 7 makes economic sense (weekly patterns?)
2. Try ARIMA(1,1,1) vs ARIMA(7,1,1) on out-of-sample RMSE
3. If improvement is marginal, keep the simpler model
4. Note the limitation in documentation

</details>

---

### Question 3.1.3
**You need to forecast crude oil prices for the next 30 days. Your ARIMA model shows the forecast as almost a flat line with widening confidence intervals. Is this a bug or expected behavior? Explain.**

<details>
<summary>Click to reveal answer</summary>

**Answer:**

**This is EXPECTED behavior, not a bug.**

**Why forecasts flatten:**

For ARIMA(p,d,q), long-horizon forecasts converge to:
- **Stationary ARIMA (d=0):** Converges to the unconditional mean
- **Integrated ARIMA (d=1):** Converges to a constant or linear trend

**Mathematical reason:**
With d=1, your model essentially says:
```
Future change = φ × Recent change + θ × Recent error + New error
```
As you forecast further out:
- "Recent change" becomes yesterday's forecast (not actual data)
- The AR/MA effects decay (|φ| < 1, |θ| < 1)
- Eventually, expected change → 0
- Forecast = last known price + small drift

**Why confidence intervals widen:**
- Each step adds forecast uncertainty
- Errors compound: σ²(h) = σ² × (1 + ψ₁² + ψ₂² + ... + ψ_{h-1}²)
- By day 30, you've accumulated 30 days of uncertainty

**For crude oil specifically:**
- Oil prices are close to random walk
- Best prediction of future price ≈ current price
- Flat forecast IS the honest answer!

**If you want dynamic forecasts:**
1. **Rolling forecast** — re-estimate daily with new data
2. **Scenario modeling** — overlay specific assumptions (OPEC cuts, demand shocks)
3. **Different model class** — regime-switching models, machine learning with external features

**The flat line teaches you:** ARIMA is honest about what it doesn't know. It's not trying to predict unpredictable shocks.

</details>

---

## Section 3.2: Code Challenges

### Challenge 3.2.1
**Write code to perform KPSS test alongside ADF test for robust stationarity determination. Explain the decision logic.**

<details>
<summary>Click to reveal answer</summary>

```python
from statsmodels.tsa.stattools import adfuller, kpss

def robust_stationarity_test(timeseries, name="Series"):
    """
    Performs both ADF and KPSS tests for robust stationarity determination.
    
    Decision logic:
    - ADF tests H0: unit root exists (non-stationary)
    - KPSS tests H0: series is stationary
    
    Combined interpretation:
    | ADF Result | KPSS Result | Conclusion |
    |------------|-------------|------------|
    | Reject H0  | Fail reject | Stationary |
    | Fail reject| Reject H0   | Non-stationary |
    | Both reject| -           | Trend stationary |
    | Neither    | -           | Inconclusive |
    """
    print(f"\n{'='*50}")
    print(f"Stationarity Tests for: {name}")
    print('='*50)
    
    # ADF Test
    adf_result = adfuller(timeseries, autolag='AIC')
    adf_pvalue = adf_result[1]
    adf_stationary = adf_pvalue <= 0.05
    
    print(f"\nADF Test:")
    print(f"  Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_pvalue:.4f}")
    print(f"  Result: {'Stationary' if adf_stationary else 'Non-stationary'}")
    
    # KPSS Test
    # Note: KPSS can raise warnings for large datasets
    kpss_result = kpss(timeseries, regression='c', nlags='auto')
    kpss_pvalue = kpss_result[1]
    kpss_stationary = kpss_pvalue > 0.05  # Note: opposite logic!
    
    print(f"\nKPSS Test:")
    print(f"  Statistic: {kpss_result[0]:.4f}")
    print(f"  p-value: {kpss_pvalue:.4f}")
    print(f"  Result: {'Stationary' if kpss_stationary else 'Non-stationary'}")
    
    # Combined Decision
    print(f"\n{'='*50}")
    print("COMBINED DECISION:")
    
    if adf_stationary and kpss_stationary:
        print("✓ STATIONARY (both tests agree)")
        return 0  # d=0
    elif not adf_stationary and not kpss_stationary:
        print("✗ NON-STATIONARY (both tests agree) → Differencing needed")
        return 1  # d=1 (at minimum)
    elif adf_stationary and not kpss_stationary:
        print("? TREND STATIONARY (conflicting) → Consider detrending")
        return 0  # or use trend term
    else:  # not adf_stationary and kpss_stationary
        print("? INCONCLUSIVE (conflicting) → Be cautious, try d=1")
        return 1
    
# Usage:
d_suggested = robust_stationarity_test(df['Close'], "Raw Crude Oil")
d_suggested = robust_stationarity_test(df['Close'].diff().dropna(), "Differenced Oil")
```

**Why use both tests:**
- ADF has low power against near-unit-root alternatives
- KPSS complements by testing the opposite hypothesis
- Agreement between both tests gives more confidence

</details>

---

### Challenge 3.2.2
**Your current RMSE is in dollars ($6.88). Write code to also calculate MAPE (Mean Absolute Percentage Error) and explain why MAPE might be more useful for crude oil.**

<details>
<summary>Click to reveal answer</summary>

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def comprehensive_forecast_evaluation(actual, predicted, model_name="Model"):
    """
    Calculates multiple error metrics for forecast evaluation.
    
    Parameters:
    -----------
    actual : array-like
        Actual observed values
    predicted : array-like
        Model predictions
    model_name : str
        Name for display purposes
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # SMAPE (Symmetric MAPE) - more robust
    smape = np.mean(2 * np.abs(actual - predicted) / 
                    (np.abs(actual) + np.abs(predicted))) * 100
    
    # Direction Accuracy (did we predict up/down correctly?)
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    print(f"\n{'='*50}")
    print(f"Forecast Evaluation: {model_name}")
    print('='*50)
    print(f"RMSE:  ${rmse:.2f}")
    print(f"       (Average error magnitude in dollars)")
    print(f"\nMAE:   ${mae:.2f}")
    print(f"       (Median-like error, less sensitive to outliers)")
    print(f"\nMAPE:  {mape:.2f}%")
    print(f"       (Percentage error - scale independent)")
    print(f"\nSMAPE: {smape:.2f}%")
    print(f"       (Symmetric percentage error)")
    print(f"\nDirection Accuracy: {direction_accuracy:.1f}%")
    print(f"       (% of times up/down direction was correct)")
    print(f"       (Random guess = 50%)")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'direction_accuracy': direction_accuracy
    }

# Usage:
metrics = comprehensive_forecast_evaluation(test, forecast_series, "ARIMA(1,1,1)")
```

**Why MAPE is more useful for crude oil:**

1. **Scale independence:** 
   - RMSE of $6.88 means nothing without context
   - MAPE of 9% immediately tells you "predictions are off by about 9%"

2. **Comparability across time:**
   - In 2020, oil was $20 → RMSE $6.88 is huge (34% error)
   - In 2024, oil is $75 → RMSE $6.88 is acceptable (9% error)
   - MAPE normalizes this automatically

3. **Industry standard:**
   - Energy forecasters typically report MAPE
   - Academic papers on oil price forecasting use MAPE for comparison
   - Allows benchmarking against other models/papers

**MAPE limitations:**
- Undefined when actual = 0 (not an issue for oil prices)
- Asymmetric: 50% over-prediction and 50% under-prediction give different MAPE
- That's why SMAPE (symmetric) is included as alternative

</details>

---

### Challenge 3.2.3
**Add an ARCH-LM test to your diagnostics to check for volatility clustering. Interpret what the result means.**

<details>
<summary>Click to reveal answer</summary>

```python
from statsmodels.stats.diagnostic import het_arch

def complete_residual_diagnostics(residuals, lags=10):
    """
    Comprehensive residual diagnostic tests for ARIMA models.
    """
    print("\n" + "="*60)
    print("RESIDUAL DIAGNOSTICS")
    print("="*60)
    
    # 1. Ljung-Box Test (autocorrelation in levels)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_pvalue = lb_result['lb_pvalue'].values[0]
    
    print(f"\n1. LJUNG-BOX TEST (Autocorrelation)")
    print(f"   H0: Residuals are white noise (no autocorrelation)")
    print(f"   p-value: {lb_pvalue:.4f}")
    if lb_pvalue > 0.05:
        print("   ✓ PASS: Residuals appear to be white noise")
    else:
        print("   ✗ FAIL: Residuals have autocorrelation - model incomplete")
    
    # 2. ARCH-LM Test (autocorrelation in squared residuals)
    arch_result = het_arch(residuals, nlags=lags)
    arch_lm_stat = arch_result[0]
    arch_pvalue = arch_result[1]
    
    print(f"\n2. ARCH-LM TEST (Volatility Clustering)")
    print(f"   H0: No ARCH effects (constant variance)")
    print(f"   LM Statistic: {arch_lm_stat:.4f}")
    print(f"   p-value: {arch_pvalue:.4f}")
    if arch_pvalue > 0.05:
        print("   ✓ PASS: No volatility clustering detected")
        print("   → ARIMA alone is sufficient")
    else:
        print("   ✗ FAIL: Volatility clustering present!")
        print("   → Consider ARIMA-GARCH model for better uncertainty estimates")
    
    # 3. Jarque-Bera Test (normality)
    from scipy import stats
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    
    print(f"\n3. JARQUE-BERA TEST (Normality)")
    print(f"   H0: Residuals are normally distributed")
    print(f"   p-value: {jb_pvalue:.4f}")
    if jb_pvalue > 0.05:
        print("   ✓ PASS: Residuals appear normally distributed")
    else:
        print("   ✗ FAIL: Non-normal residuals (fat tails likely)")
        print("   → Common for crude oil; consider t-distribution in GARCH")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    all_pass = lb_pvalue > 0.05 and arch_pvalue > 0.05 and jb_pvalue > 0.05
    if all_pass:
        print("All diagnostics pass - model is well-specified")
    else:
        print("Some diagnostics failed - see recommendations above")
    print("="*60)
    
    return {
        'ljung_box_pvalue': lb_pvalue,
        'arch_lm_pvalue': arch_pvalue,
        'jarque_bera_pvalue': jb_pvalue
    }

# Usage:
residuals = model_fit.resid
diagnostics = complete_residual_diagnostics(residuals, lags=10)
```

**Interpretation of ARCH-LM test:**

**What it tests:**
Regresses squared residuals on lagged squared residuals:
```
ε²_t = α₀ + α₁ε²_{t-1} + α₂ε²_{t-2} + ... + u_t
```
Tests if α₁ = α₂ = ... = 0

**If p-value < 0.05 (FAIL):**
- Squared residuals are autocorrelated
- Big errors follow big errors; small follow small
- Your confidence intervals are WRONG
- Periods of high volatility are underestimated
- Periods of low volatility are overestimated

**For crude oil:**
Almost certainly you'll see ARCH effects because:
1. OPEC meetings create volatility clusters
2. Geopolitical events cause prolonged uncertainty
3. Market stress propagates

**Solution:**
```python
# ARIMA-GARCH model
from arch import arch_model

# First get ARIMA residuals
arima_model = ARIMA(train, order=(1,1,1)).fit()
arima_residuals = arima_model.resid

# Then fit GARCH on residuals
garch = arch_model(arima_residuals, vol='Garch', p=1, q=1)
garch_fit = garch.fit(disp='off')

# Now you have proper volatility forecasts
```

</details>

---

# PART 4: REFLECTION QUESTIONS

Answer these in your own words to consolidate understanding:

1. **Why is crude oil particularly difficult for ARIMA to forecast compared to, say, monthly electricity demand?**

2. **If your model has MAPE of 8% and direction accuracy of 52%, should you trade based on it? Why or why not?**

3. **Your friend says "ARIMA is useless for oil because it just predicts a flat line." How do you respond?**

4. **When would you choose SARIMA over ARIMA for crude oil? What evidence would convince you?**

5. **List 3 things ARIMA cannot capture that matter for oil price movements.**

---

# FINAL CHECKLIST

Before you consider yourself proficient, make sure you can:

## Conceptual
- [ ] Explain stationarity to a non-technical person
- [ ] Draw ACF/PACF patterns for AR(1), MA(1), and ARMA(1,1)
- [ ] Explain why differencing converts non-stationary to stationary
- [ ] Interpret all parts of ARIMA model summary output

## Technical
- [ ] Perform ADF and KPSS tests and interpret jointly
- [ ] Write a complete ARIMA workflow from data loading to diagnostics
- [ ] Implement rolling forecast from scratch
- [ ] Use auto_arima with appropriate parameters

## Practical
- [ ] Know when ARIMA is appropriate vs. alternatives
- [ ] Diagnose model problems from residual plots
- [ ] Compare models using AIC, BIC, and out-of-sample RMSE
- [ ] Recognize volatility clustering and know the solution

---

**You've built a solid foundation. The next steps in your learning path should be:**
1. GARCH for volatility modeling
2. SARIMA for seasonal data
3. VAR for multivariate relationships (oil + USD + stocks)
4. Regime-switching models for structural breaks