"""
utils.py
Utility functions for ease of plotting
"""

import gmr
import matplotlib.pyplot as plt
import numpy as np

from .helpers import derivative
from .lasa import load_lasa
from .models import LWR, RBFN, LeastSquares


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


def load_data_ax(letter, show_plot=False, ax=None):
    """
    gets the trajectories coresponding to the given letter

    params:
      letter: character in ["c","j","s"]
      show_plot: whether to show the plot
      ax: axis to plot on

    returns:
      data: array of shape (number of trajectories,number of timesteps,2)
      x: array of shape(number of trajectories*number of timesteps,2)
      xd: array of shape(number of trajectories*number of timesteps,2)

    """
    letter2id = dict(c=2, j=6, s=24)
    assert letter.lower() in letter2id
    _, x, _, _, _, _ = load_lasa(letter2id[letter.lower()])
    xd = derivative(x)
    if show_plot:
        if ax is None:
            plt.figure(figsize=(10, 5))
        else:
            ax = ax
        plot_curves_ax(ax, x)
    data = x
    x = x.reshape(-1, 2)
    xd = xd.reshape(-1, 2)
    plt.show()
    return data, x, xd


def init_gaussians_ax(y, n=3, show_plot=False, ax=None):
    """
    Initializes the Gaussians based on time fragmentation.

    Params:
        y : array of shape (number of trajectories, number of timesteps, 2)
        n : Number of Gaussians
        ax : Optional axis for plotting (default: None)

    Returns:
        mvns: List of gmr.MVN() initialized objects
    """
    l = y.shape[1] // n
    y_split = [
        y[:, i * l :] if i == n - 1 else y[:, i * l : (i + 1) * l] for i in range(n)
    ]
    mvns = [gmr.MVN().from_samples(x.reshape(-1, 2)) for x in y_split]

    if show_plot or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        plot_curves_ax(ax, y)
        for mvn in mvns:
            gmr.plot_error_ellipse(ax, mvn, factors=[1])

    return mvns


def fit_least_squares(dataset, lam=1e-2, bias=False):
    # load data
    data, x, xd = load_data_ax(dataset)

    # fitting the model to data
    model = LeastSquares(lam=lam, bias=bias)
    model.fit(x, xd)

    # starting point for imitation
    x0 = data[6][0]
    x_rk4, t_tk4 = model.imitate(x0, t_end=10)

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    # plots for generated trajectory
    plot_curves_ax(axes[0], data, alpha=0.3, c="g", label="demonstrations")
    plot_curves_ax(
        axes[0], x_rk4[None], show_start_end=False, label="generated trajectory"
    )

    # vector field plot using stream line
    plot_curves_ax(axes[1], data, alpha=0.5, c="b", label="demonstrations")
    streamplot_ax(
        axes[1],
        model.predict,
        x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
        y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
        width=3,
        color="g",
    )


def _model_imitate(data, model, n=10, starting=0):
    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    mvns = init_gaussians_ax(data, n, ax=axes[0])

    # starting point for imitation
    x0 = data[starting][0]
    x_rk4, t_tk4 = model.imitate(x0, t_end=10)

    # plots for generated trajectory
    plot_curves_ax(axes[1], data, alpha=0.3, c="g", label="demonstrations")
    plot_curves_ax(
        axes[1], x_rk4[None], show_start_end=False, label="generated trajectory"
    )
    axes[1].set_title("Generated Trajectory")

    # vector field plot using stream line
    plot_curves_ax(axes[2], data, alpha=0.5, c="b", label="demonstrations")
    streamplot_ax(
        axes[1],
        model.predict,
        x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
        y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
        width=3,
        color="g",
    )
    axes[2].set_title("Vector Field")


def fit_lwr(dataset, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = LWR(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, model, n, starting=6)


def fit_rbfn(dataset, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, model, n, starting=0)


def _get_model(model, mvns, bias):
    if model == "lwr":
        model = LWR(mvns, bias=bias)
    elif model == "rbfn":
        model = RBFN(mvns, bias=bias)
    else:
        raise ValueError("Model should be either 'lwr' or 'rbfn'")


def _different_initial_points(dataset, model, initial_points, n=4, bias=False):
    """
    Generates kx2 plots for different starting points.

    Params:
        dataset: dataset to use
        initial_points: array of k starting points
        n: number of Gaussians
        bias: whether to include bias in RBFN
    """
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)

    model = _get_model(model, mvns, bias)
    model.fit(x, xd)

    k = len(initial_points)
    fig, axes = plt.subplots(k, 2, figsize=(10, 5 * k))

    for i, start_idx in enumerate(initial_points):
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


def different_initial_points_lwr(dataset, initial_points, n=4, bias=False):
    _different_initial_points(
        dataset=dataset,
        model="lwr",
        initial_points=initial_points,
        n=n,
        bias=bias,
    )


def different_initial_points_rbfn(dataset, initial_points, n=4, bias=False):
    _different_initial_points(
        dataset=dataset,
        model="rbfn",
        initial_points=initial_points,
        n=n,
        bias=bias,
    )


def _generalisation(dataset, model, n_values, bias=False):
    """
    Generates len(n_values)x2 plots for different numbers of Gaussians.

    Params:
        dataset: dataset to use
        n_values: array of different values for number of Gaussians
        bias: whether to include bias in RBFN
    """
    data, x, xd = load_data_ax(dataset)
    k = len(n_values)
    fig, axes = plt.subplots(k, 3, figsize=(10, 5 * k))

    for i, n in enumerate(n_values):
        mvns = init_gaussians_ax(data, n, ax=axes[i, 0])
        axes[i, 0].set_title(f"Gaussians with {n} components")

        model = _get_model(model, mvns, bias)
        model.fit(x, xd)

        x0 = data[0][0]
        x_rk4, _ = model.imitate(x0, t_end=10)

        plot_curves_ax(axes[i, 1], data, alpha=0.3, c="g", label="demonstrations")
        plot_curves_ax(
            axes[i, 1], x_rk4[None], show_start_end=False, label="generated trajectory"
        )
        axes[i, 1].set_title(f"Trajectory for {n} Gaussians")

        plot_curves_ax(axes[i, 2], data, alpha=0.5, c="b", label="demonstrations")
        streamplot_ax(
            axes[i, 2],
            model.predict,
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
        )
        axes[i, 2].set_title(f"Vector field for {n} Gaussians")

    plt.tight_layout()
    plt.show()


def generalisation_lwr(dataset, n_values, bias=False):
    _generalisation(
        dataset=dataset,
        model="lwr",
        n_values=n_values,
        bias=bias,
    )


def generalisation_rbfn(dataset, n_values, bias=False):
    _generalisation(
        dataset=dataset,
        model="rbfn",
        n_values=n_values,
        bias=bias,
    )
