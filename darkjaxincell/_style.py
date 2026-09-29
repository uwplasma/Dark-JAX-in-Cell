"""A white plotting canvas with restrained, legible ghost colours."""

from contextlib import contextmanager


@contextmanager
def midnight():
    """Use the parent's figure typography with a white canvas and dark accents."""
    import matplotlib as mpl
    from jaxincell import style

    with mpl.rc_context():
        style()
        mpl.rcParams.update({
            "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
            "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
            "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
            "xtick.major.size": 5, "ytick.major.size": 5,
            "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
            "xtick.minor.size": 3, "ytick.minor.size": 3,
            "lines.linewidth": 2, "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "text.color": "#202124",
            "axes.labelcolor": "#202124", "axes.edgecolor": "#30343B",
            "xtick.color": "#202124", "ytick.color": "#202124", "grid.color": "#D8DEE8",
            "axes.prop_cycle": mpl.cycler(color=[
                "#6A3D9A", "#0072B2", "#D55E00", "#009E73", "#B03568"]),
        })
        yield
