"""
theme.py

Shared Bloomberg Terminal-style theme constants and layout helper.
Used by both app.py (US dashboard) and dashboard_jp.py (JP dashboard).
"""
import copy

BLOOMBERG_LAYOUT = dict(
    paper_bgcolor="#0b0f14",
    plot_bgcolor="#0e1419",
    font=dict(family="IBM Plex Mono, Consolas, Courier New, monospace", size=11, color="#8899aa"),
    title_font=dict(size=13, color="#ff9800"),
    xaxis=dict(
        gridcolor="#1a2332", gridwidth=1, griddash="dot",
        zerolinecolor="#1a2332", zerolinewidth=1,
        tickfont=dict(size=10, color="#5a6a7a"),
        title_font=dict(size=11, color="#5a6a7a"),
        showline=True, linecolor="#1a2332", linewidth=1,
    ),
    yaxis=dict(
        gridcolor="#1a2332", gridwidth=1, griddash="dot",
        zerolinecolor="#1a2332", zerolinewidth=1,
        tickfont=dict(size=10, color="#5a6a7a"),
        title_font=dict(size=11, color="#5a6a7a"),
        showline=True, linecolor="#1a2332", linewidth=1,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor="#1a2332", borderwidth=1,
        font=dict(size=10, color="#8899aa"),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#111820", bordercolor="#1a2332",
        font=dict(family="IBM Plex Mono, Consolas, monospace", size=11, color="#c8cdd3"),
    ),
    margin=dict(t=35, b=50, l=50, r=20),
)

# Bloomberg-style color constants
BB_CYAN = "#00b4d8"
BB_AMBER = "#ff9800"
BB_GREEN = "#00c853"
BB_RED = "#ff1744"
BB_BLUE = "#3a86ff"
BB_GREY = "#5a6a7a"
BB_LIGHT = "#c8cdd3"
BB_DIM = "#2a3a4a"


def bb_layout(**overrides):
    """Merge BLOOMBERG_LAYOUT with overrides (deep-merge for nested dicts)."""
    layout = copy.deepcopy(BLOOMBERG_LAYOUT)
    for k, v in overrides.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k].update(v)
        else:
            layout[k] = v
    return layout
