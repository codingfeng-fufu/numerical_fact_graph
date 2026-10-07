from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Mapping


@dataclass(frozen=True)
class ForecastResult:
    prediction: float
    method: str
    selected_method: str
    diagnostics: dict[str, float | int | str]


DEFAULT_FORECAST_SELECTOR_CONFIG: dict[str, object] = {
    "name": "recent_growth_risk_t0.22_w0.5",
    "selector_type": "linear_naive_rules",
    "base_weight": 0.92,
    "short_history_n": 1,
    "sign_change_min": 999,
    "volatility_threshold": 1.0e12,
    "extrapolation_threshold": 1.0e12,
    "recent_growth_threshold": 0.22,
    "recent_growth_weight": 0.5,
    "floor_at_zero": True,
}

FORECAST_PROFILES: dict[str, dict[str, object]] = {
    "linear": {"selector_type": "method", "method": "linear", "name": "linear"},
    "naive_last": {"selector_type": "method", "method": "naive_last", "name": "naive_last"},
    "auto": {"selector_type": "method", "method": "auto", "name": "auto"},
    str(DEFAULT_FORECAST_SELECTOR_CONFIG["name"]): dict(DEFAULT_FORECAST_SELECTOR_CONFIG),
    "default": dict(DEFAULT_FORECAST_SELECTOR_CONFIG),
    "risk_aware_default": dict(DEFAULT_FORECAST_SELECTOR_CONFIG),
}


def forecast_profile(name: str | None = None) -> dict[str, object]:
    profile_name = (name or "default").strip()
    if profile_name not in FORECAST_PROFILES:
        raise KeyError(
            f"Unknown forecast profile '{profile_name}'. "
            f"Available profiles: {sorted(FORECAST_PROFILES)}"
        )
    return dict(FORECAST_PROFILES[profile_name])


def forecast_series(
    year_values: dict[int, float],
    horizon: int,
    *,
    method: str = "auto",
    selector_config: Mapping[str, object] | None = None,
) -> ForecastResult:
    if len(year_values) < 2:
        raise ValueError(f"forecast needs >=2 historical data points; got {len(year_values)}")

    xs = sorted(year_values)
    ys = [float(year_values[x]) for x in xs]
    canonical = _canonical_method(method)
    if selector_config is not None and canonical in {"auto", "configured_auto", "selector"}:
        candidate = _configured_selector(xs, ys, horizon, selector_config)
        canonical = "configured_auto"
    elif canonical in {"configured_auto", "selector"}:
        candidate = _auto_forecast(xs, ys, horizon)
        canonical = "auto"
    elif canonical == "auto":
        candidate = _auto_forecast(xs, ys, horizon)
    else:
        candidate = _method_forecast(canonical, xs, ys, horizon)
    return ForecastResult(
        prediction=float(candidate["prediction"]),
        method=canonical,
        selected_method=str(candidate["selected_method"]),
        diagnostics=dict(candidate["diagnostics"]),
    )


def _canonical_method(method: str | None) -> str:
    normalized = (method or "auto").strip().lower().replace("-", "_")
    aliases = {
        "linear_regression": "linear",
        "linear_trend": "linear",
        "last": "naive_last",
        "last_value": "naive_last",
        "naive": "naive_last",
        "median": "median_growth",
        "rolling": "rolling_growth",
        "log": "log_linear",
    }
    return aliases.get(normalized, normalized)


def _method_forecast(method: str, xs: list[int], ys: list[float], horizon: int) -> dict[str, object]:
    if method == "naive_last":
        prediction = ys[-1]
    elif method == "linear":
        prediction = _linear(xs, ys, horizon)
    elif method == "cagr":
        prediction = _cagr(xs, ys, horizon)
    elif method == "rolling_growth":
        prediction = _rolling_growth(ys)
    elif method == "median_growth":
        prediction = _median_growth(ys)
    elif method == "log_linear":
        prediction = _log_linear(xs, ys, horizon)
    else:
        raise ValueError(f"Unsupported forecast method: {method}")
    return {
        "prediction": prediction,
        "selected_method": method,
        "diagnostics": _series_diagnostics(xs, ys, prediction),
    }


def _configured_selector(
    xs: list[int],
    ys: list[float],
    horizon: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    selector_type = str(config.get("selector_type", "linear_naive_rules"))
    if selector_type == "method":
        method = _canonical_method(str(config.get("method", "linear")))
        return _method_forecast(method, xs, ys, horizon)
    if selector_type != "linear_naive_rules":
        raise ValueError(f"Unsupported selector_type: {selector_type}")

    linear = _linear(xs, ys, horizon)
    naive = ys[-1]
    diagnostics = _series_diagnostics(xs, ys, linear)
    weight = _float_config(config, "base_weight", 1.0)
    triggered: list[str] = []

    if len(ys) <= int(config.get("short_history_n", 2)):
        weight = min(weight, _float_config(config, "short_history_weight", weight))
        triggered.append("short_history")
    if int(diagnostics["growth_sign_changes"]) >= int(config.get("sign_change_min", 1)):
        weight = min(weight, _float_config(config, "sign_change_weight", weight))
        triggered.append("sign_change")
    if float(diagnostics["growth_volatility"]) > _float_config(config, "volatility_threshold", float("inf")):
        weight = min(weight, _float_config(config, "volatile_weight", weight))
        triggered.append("volatile")
    if float(diagnostics["extrapolation_ratio"]) > _float_config(config, "extrapolation_threshold", float("inf")):
        weight = min(weight, _float_config(config, "extrapolation_weight", weight))
        triggered.append("extrapolation")
    growth_rates = _growth_rates(ys)
    recent_growth = growth_rates[-1] if growth_rates else 0.0
    if recent_growth > _float_config(config, "recent_growth_threshold", float("inf")):
        weight = min(weight, _float_config(config, "recent_growth_weight", weight))
        triggered.append("recent_growth")

    weight = min(max(weight, 0.0), 1.0)
    prediction = weight * linear + (1.0 - weight) * naive
    if bool(config.get("floor_at_zero", True)):
        prediction = max(0.0, prediction)

    diagnostics.update({
        "linear_prediction": float(linear),
        "naive_last_prediction": float(naive),
        "linear_weight": float(weight),
        "recent_growth": float(recent_growth),
        "triggered_rules": ",".join(triggered) if triggered else "none",
    })
    return {
        "prediction": prediction,
        "selected_method": f"linear_naive_rules:w={weight:.3f}",
        "diagnostics": diagnostics,
    }


def _auto_forecast(xs: list[int], ys: list[float], horizon: int) -> dict[str, object]:
    linear = _linear(xs, ys, horizon)
    naive = ys[-1]
    med = _median_growth(ys)
    log_linear = _log_linear(xs, ys, horizon)
    growth_rates = _growth_rates(ys)
    diagnostics = _series_diagnostics(xs, ys, linear)

    if len(ys) <= 2:
        selected = "short_history_linear_shrink"
        prediction = 0.6 * linear + 0.4 * naive
    elif int(diagnostics["growth_sign_changes"]) >= 1:
        selected = "volatile_linear_naive_median_blend"
        prediction = 0.5 * linear + 0.3 * naive + 0.2 * med
    elif float(diagnostics["growth_volatility"]) > 0.10:
        selected = "volatile_linear_naive_blend"
        prediction = 0.6 * linear + 0.4 * naive
    elif growth_rates and (all(g > 0 for g in growth_rates) or all(g < 0 for g in growth_rates)):
        selected = "stable_trend_linear_log_blend"
        prediction = 0.7 * linear + 0.3 * log_linear
    else:
        selected = "linear_with_naive_shrink"
        prediction = 0.8 * linear + 0.2 * naive

    diagnostics.update({
        "linear_prediction": float(linear),
        "naive_last_prediction": float(naive),
        "median_growth_prediction": float(med),
        "log_linear_prediction": float(log_linear),
    })
    return {
        "prediction": prediction,
        "selected_method": selected,
        "diagnostics": diagnostics,
    }


def _linear(xs: list[int], ys: list[float], horizon: int) -> float:
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    den = sum((x - x_bar) ** 2 for x in xs)
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True)) / den if abs(den) > 1e-12 else 0.0
    intercept = y_bar - slope * x_bar
    return intercept + slope * horizon


def _cagr(xs: list[int], ys: list[float], horizon: int) -> float:
    if ys[0] <= 0:
        return ys[-1]
    periods = max(xs[-1] - xs[0], 1)
    growth = (ys[-1] / ys[0]) ** (1.0 / periods) - 1.0
    return ys[-1] * ((1.0 + growth) ** max(horizon - xs[-1], 1))


def _rolling_growth(ys: list[float]) -> float:
    rates = _growth_rates(ys)
    if not rates:
        return ys[-1]
    recent = rates[-3:]
    weights = list(range(1, len(recent) + 1))
    growth = sum(weight * rate for weight, rate in zip(weights, recent, strict=True)) / sum(weights)
    return ys[-1] * (1.0 + growth)


def _median_growth(ys: list[float]) -> float:
    rates = _growth_rates(ys)
    return ys[-1] * (1.0 + median(rates)) if rates else ys[-1]


def _log_linear(xs: list[int], ys: list[float], horizon: int) -> float:
    if any(y <= 0 for y in ys):
        return _linear(xs, ys, horizon)
    logs = [math.log(y) for y in ys]
    return math.exp(_linear(xs, logs, horizon))


def _growth_rates(ys: list[float]) -> list[float]:
    return [
        ys[i] / ys[i - 1] - 1.0
        for i in range(1, len(ys))
        if abs(ys[i - 1]) > 1e-12
    ]


def _float_config(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series_diagnostics(xs: list[int], ys: list[float], prediction: float) -> dict[str, float | int | str]:
    rates = _growth_rates(ys)
    mean_growth = sum(rates) / len(rates) if rates else 0.0
    volatility = (
        math.sqrt(sum((rate - mean_growth) ** 2 for rate in rates) / len(rates))
        if rates
        else 0.0
    )
    sign_changes = sum(1 for left, right in zip(rates, rates[1:], strict=False) if left * right < 0)
    last_value = ys[-1]
    extrapolation_ratio = abs(prediction - last_value) / max(abs(last_value), 1e-9)
    return {
        "n_history": len(ys),
        "history_start": xs[0],
        "history_end": xs[-1],
        "last_value": float(last_value),
        "mean_growth": float(mean_growth),
        "growth_volatility": float(volatility),
        "growth_sign_changes": int(sign_changes),
        "extrapolation_ratio": float(extrapolation_ratio),
    }
