"""
utils.py
Utility functions for ease of plotting
"""

import matplotlib.pyplot as plt
import numpy as np

from .helpers import init_gaussians, load_data
from .models import RBFN


def plot_curves_ax(ax, x, show_start_end=True, **kwargs):
    """
    Plots 2D curves of trajectories on a given axis.

    Params:
        ax: matplotlib axis to plot on
        x: array of shape (number of curves, n_steps_per_curve, 2)
        show_start_end: whether to label start and end points
    """
    if show_start_end:
        start_label, end_label = "start", "end"
    else:
        start_label, end_label = None, None

    for t in range(x.shape[0]):
        ax.scatter(x[t][0, 0], x[t][0, 1], c="k", label=start_label)
        ax.scatter(x[t][-1, 0], x[t][-1, 1], c="b", label=end_label)
        ax.plot(x[t][:, 0], x[t][:, 1], **kwargs)
        if t == 0:
            kwargs.pop("label", None)
            start_label, end_label = None, None

    ax.legend()


def streamplot_ax(ax, f, x_axis=(0, 100), y_axis=(0, 100), n=1000, width=1, **kwargs):
    """
    Visualizes the vector field on a given axis.

    Params:
        ax: matplotlib axis to plot on
        f: function to predict the velocities in DS
        x_axis: x axis limits
        y_axis: y axis limits
        n: number of points per axis (total n*n predictions)
        width: width of the vector
        **kwargs: additional arguments for plt.streamplot
    """
    a, b = np.linspace(x_axis[0], x_axis[1], n), np.linspace(y_axis[0], y_axis[1], n)
    X, Y = np.meshgrid(a, b)
    X_test = np.stack([X, Y], axis=-1).reshape(-1, 2)
    Y_pred = f(X_test)
    U, V = np.split(Y_pred.reshape(n, n, 2), 2, axis=-1)
    U, V = U[..., 0], V[..., 0]
    speed = np.sqrt(U**2 + V**2)
    lw = width * speed / speed.max()
    ax.streamplot(X, Y, U, V, linewidth=lw, **kwargs)


def different_starting_points_rbfn(dataset, starting_points, n=4, bias=False):
    """
    Generates kx2 plots for different starting points.

    Params:
        dataset: dataset to use
        starting_points: array of k starting points
        n: number of Gaussians
        bias: whether to include bias in RBFN
    """
    data, x, xd = load_data(dataset)
    mvns = init_gaussians(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)

    k = len(starting_points)
    fig, axes = plt.subplots(k, 2, figsize=(10, 5 * k))

    for i, start_idx in enumerate(starting_points):
        x0 = data[start_idx][0]
        x_rk4, _ = model.imitate(x0, t_end=10)

        plot_curves_ax(axes[i, 0], data, alpha=0.3, c="g", label="demonstrations")
        plot_curves_ax(
            axes[i, 0], x_rk4[None], show_start_end=False, label="generated trajectory"
        )
        axes[i, 0].set_title(f"Trajectory for starting point {start_idx}")

        plot_curves_ax(axes[i, 1], data, alpha=0.5, c="b", label="demonstrations")
        streamplot_ax(
            axes[i, 1],
            model.predict,
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
        )
        axes[i, 1].set_title(f"Vector field for starting point {start_idx}")

    plt.tight_layout()
    plt.show()
