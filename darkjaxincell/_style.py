"""A readable midnight palette for the ghost's figures."""

from contextlib import contextmanager


@contextmanager
def midnight():
    """Use a dark, accessible Matplotlib style without changing global settings."""
    import matplotlib as mpl

    with mpl.rc_context({
        "figure.facecolor": "#101018", "axes.facecolor": "#171724",
        "savefig.facecolor": "#101018", "text.color": "#eeeaf4",
        "axes.labelcolor": "#eeeaf4", "axes.edgecolor": "#8f899e",
        "xtick.color": "#ccc6d4", "ytick.color": "#ccc6d4",
        "grid.color": "#554d67", "axes.prop_cycle": mpl.cycler(color=[
            "#bd93f9", "#50fae4", "#ffb86c", "#ff79c6", "#f1fa8c"]),
    }):
        yield
