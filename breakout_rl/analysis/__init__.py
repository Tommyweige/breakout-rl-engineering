"""Reusable analysis helpers for Breakout training evidence.

The public modules are loaded lazily so NumPy-only plots do not import the
PyTorch training runtime and its platform-specific native libraries.
"""

__all__ = [
    "analyze_q_values",
    "generate_noisy_estimates",
    "infer_q_values",
    "load_probe_states",
    "plot_q_probe_summary",
    "plot_overestimation_bias",
    "run_noise_sweep",
    "save_probe_states",
    "simulate_overestimation",
    "summarize_q_values",
    "validate_probe_states",
]


def __getattr__(name: str):
    overestimation_names = {
        "generate_noisy_estimates",
        "plot_overestimation_bias",
        "run_noise_sweep",
        "simulate_overestimation",
    }
    q_value_names = {
        "analyze_q_values",
        "infer_q_values",
        "load_probe_states",
        "plot_q_probe_summary",
        "save_probe_states",
        "summarize_q_values",
        "validate_probe_states",
    }
    if name in overestimation_names:
        from breakout_rl.analysis import overestimation

        value = getattr(overestimation, name)
    elif name in q_value_names:
        from breakout_rl.analysis import q_values

        value = getattr(q_values, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
