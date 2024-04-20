from matplotlib import pyplot as plt

figsize = (3.375, 2.5)
double_figsize = (6.75, 2.5)
large_figsize = (6.75, 4.5)

markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'H', 'X']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']


def set_plot_style(fontsize=9):
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "axes.titlesize": fontsize
    }
    plt.rcParams.update(tex_fonts)

    return
