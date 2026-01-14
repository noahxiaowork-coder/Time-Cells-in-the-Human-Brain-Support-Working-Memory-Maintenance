"""Shared utility functions for statistical analysis and plotting."""

import numpy as np


def significance_stars(p_value):
    """Convert p-value to significance notation (*** / ** / * / ns)."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def add_significance_bar(ax, x1, x2, y, p_value, text_offset=0.02, linewidth=1, color='black'):
    """Draw significance bar with stars between x1 and x2."""
    bar_y = y + text_offset
    ax.plot([x1, x1, x2, x2], [y, bar_y, bar_y, y],
            color=color, linewidth=linewidth)

    sig = significance_stars(p_value)
    ax.text((x1 + x2) / 2, bar_y, sig,
            ha='center', va='bottom', fontsize=12, color=color)


def clean_region_name(region_field):
    """Remove _left/_right suffixes from brain region names."""
    reg = region_field

    if isinstance(reg, np.ndarray):
        reg = np.squeeze(reg)
        if hasattr(reg, "dtype") and reg.dtype.kind in ("U", "S"):
            reg = "".join(reg.flat)
        else:
            reg = str(reg)
    else:
        reg = str(reg)

    reg = reg.strip().lower()

    for suffix in ("_left", "_right"):
        if reg.endswith(suffix):
            reg = reg[:-len(suffix)]
            break

    return reg
