use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::VecDeque;

/// Calculate Simple Moving Average (SMA) for a given window
fn calculate_sma(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    let mut queue: VecDeque<f64> = VecDeque::with_capacity(window);

    for (i, &value) in data.iter().enumerate() {
        queue.push_back(value);
        sum += value;

        if queue.len() > window {
            if let Some(old) = queue.pop_front() {
                sum -= old;
            }
        }

        if i >= window - 1 {
            result.push(sum / window as f64);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Calculate Exponentially Weighted Moving Average (EWMA)
fn calculate_ewma(data: &[f64], span: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ewma = data[0];

    result.push(ewma);

    for &value in data.iter().skip(1) {
        ewma = alpha * value + (1.0 - alpha) * ewma;
        result.push(ewma);
    }

    result
}

/// Calculate Rolling Maximum
fn calculate_rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let start = if i >= window - 1 { i - window + 1 } else { 0 };
        let slice = &data[start..=i];

        let max_val = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if i >= window - 1 {
            result.push(max_val);
        } else {
            result.push(f64::NAN);
        }
    }

    result
}

/// Calculate Rolling Standard Deviation
fn calculate_rolling_std(data: &[f64], window: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        if i < window - 1 {
            result.push(f64::NAN);
            continue;
        }

        let start = i - window + 1;
        let slice = &data[start..=i];

        // Calculate mean
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;

        // Calculate variance
        let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;

        result.push(variance.sqrt());
    }

    result
}

/// Calculate daily average standard deviation (for 24-hour periods)
fn calculate_daily_avg_std(volumes: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(volumes.len());
    const HOURS_PER_DAY: usize = 24;

    for i in 0..volumes.len() {
        if i < HOURS_PER_DAY - 1 {
            result.push(f64::NAN);
            continue;
        }

        let start = i - HOURS_PER_DAY + 1;
        let slice = &volumes[start..=i];

        // Calculate mean
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;

        // Calculate standard deviation
        let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;

        result.push(variance.sqrt());
    }

    result
}

/// Detect pump-and-dump anomalies using the "Best Setting" algorithm
///
/// Parameters:
/// - opens: Vec of open prices
/// - highs: Vec of high prices
/// - volumes: Vec of volume data
///
/// Returns: Vec of indices where pump-and-dump anomalies were detected
#[pyfunction]
fn detect_anomalies(opens: Vec<f64>, highs: Vec<f64>, volumes: Vec<f64>) -> PyResult<Vec<usize>> {
    let n = opens.len();

    if n != highs.len() || n != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input vectors must have the same length",
        ));
    }

    // Constants from paper
    const W_SHORT: usize = 12; // 12 hours
    const W_MEDIUM: usize = 480; // 20 days (20 * 24)
    const W_LONG: usize = 720; // 30 days (30 * 24)
    const PRICE_THRESHOLD: f64 = 1.90; // 90% increase = 1.9x
    const VOLUME_THRESHOLD: f64 = 5.0; // 400% increase = 5x

    // Calculate indicators
    let ma_open_12h = calculate_sma(&opens, W_SHORT);
    let ma_volume_12h = calculate_sma(&volumes, W_SHORT);
    let ewma_volume_20d = calculate_ewma(&volumes, W_MEDIUM);
    let max_volume_30d = calculate_rolling_max(&volumes, W_LONG);
    let daily_std_volume = calculate_daily_avg_std(&volumes);

    let mut anomalies = Vec::new();

    // Start checking after we have enough data for all windows
    let start_idx = W_LONG.max(W_MEDIUM).max(W_SHORT);

    for i in start_idx..n {
        // Skip if any indicator is NaN
        if ma_open_12h[i].is_nan()
            || ma_volume_12h[i].is_nan()
            || ewma_volume_20d[i].is_nan()
            || max_volume_30d[i].is_nan()
            || daily_std_volume[i].is_nan()
        {
            continue;
        }

        // Condition 1: Price Spike
        // High[i] > 1.9 * MA(Open, 12h)
        let price_condition = highs[i] > PRICE_THRESHOLD * ma_open_12h[i];

        // Condition 2: Base Volume Spike
        // Volume[i] > 5 * MA(Volume, 12h)
        let base_volume_condition = volumes[i] > VOLUME_THRESHOLD * ma_volume_12h[i];

        // Condition 3: Volatility & Noise Filter (Equation 4 from paper)
        // Volume[i] > EWMA(Volume, 20d) + 2 * σ_daily(Volume)
        // AND
        // Volume[i] < MAX(Volume, 30d)
        let volatility_lower = volumes[i] > ewma_volume_20d[i] + 2.0 * daily_std_volume[i];
        let volatility_upper = volumes[i] < max_volume_30d[i];

        let volatility_condition = volatility_lower && volatility_upper;

        // All conditions must be met
        if price_condition && base_volume_condition && volatility_condition {
            anomalies.push(i);
        }
    }

    Ok(anomalies)
}

/// Detect anomalies and return detailed metrics for each detection
#[pyfunction]
fn detect_anomalies_with_metrics(
    opens: Vec<f64>,
    highs: Vec<f64>,
    volumes: Vec<f64>,
) -> PyResult<Vec<(usize, f64, f64)>> {
    let n = opens.len();

    if n != highs.len() || n != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Input vectors must have the same length",
        ));
    }

    const W_SHORT: usize = 12;
    const W_MEDIUM: usize = 480;
    const W_LONG: usize = 720;
    const PRICE_THRESHOLD: f64 = 1.90;
    const VOLUME_THRESHOLD: f64 = 5.0;

    let ma_open_12h = calculate_sma(&opens, W_SHORT);
    let ma_volume_12h = calculate_sma(&volumes, W_SHORT);
    let ewma_volume_20d = calculate_ewma(&volumes, W_MEDIUM);
    let max_volume_30d = calculate_rolling_max(&volumes, W_LONG);
    let daily_std_volume = calculate_daily_avg_std(&volumes);

    let mut anomalies = Vec::new();
    let start_idx = W_LONG.max(W_MEDIUM).max(W_SHORT);

    for i in start_idx..n {
        if ma_open_12h[i].is_nan()
            || ma_volume_12h[i].is_nan()
            || ewma_volume_20d[i].is_nan()
            || max_volume_30d[i].is_nan()
            || daily_std_volume[i].is_nan()
        {
            continue;
        }

        let price_condition = highs[i] > PRICE_THRESHOLD * ma_open_12h[i];
        let base_volume_condition = volumes[i] > VOLUME_THRESHOLD * ma_volume_12h[i];
        let volatility_lower = volumes[i] > ewma_volume_20d[i] + 2.0 * daily_std_volume[i];
        let volatility_upper = volumes[i] < max_volume_30d[i];
        let volatility_condition = volatility_lower && volatility_upper;

        if price_condition && base_volume_condition && volatility_condition {
            // Calculate spike percentages
            let price_spike_pct = ((highs[i] / ma_open_12h[i]) - 1.0) * 100.0;
            let volume_spike_pct = ((volumes[i] / ma_volume_12h[i]) - 1.0) * 100.0;

            anomalies.push((i, price_spike_pct, volume_spike_pct));
        }
    }

    Ok(anomalies)
}

/// Calculate all technical indicators (for debugging/visualization)
#[pyfunction]
fn calculate_indicators(
    opens: Vec<f64>,
    volumes: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    const W_SHORT: usize = 12;
    const W_MEDIUM: usize = 480;
    const W_LONG: usize = 720;

    let ma_open_12h = calculate_sma(&opens, W_SHORT);
    let ma_volume_12h = calculate_sma(&volumes, W_SHORT);
    let ewma_volume_20d = calculate_ewma(&volumes, W_MEDIUM);
    let max_volume_30d = calculate_rolling_max(&volumes, W_LONG);
    let daily_std_volume = calculate_daily_avg_std(&volumes);

    Ok((
        ma_open_12h,
        ma_volume_12h,
        ewma_volume_20d,
        max_volume_30d,
        daily_std_volume,
    ))
}

/// Python module
#[pymodule]
fn quant_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_anomalies, m)?)?;
    m.add_function(wrap_pyfunction!(detect_anomalies_with_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_indicators, m)?)?;
    Ok(())
}
