"""
utils.py
Utility functions for ease of plotting
"""

import gmr
import matplotlib.pyplot as plt
import numpy as np

from .helpers import derivative
from .lasa import load_lasa
from .models import GMR, GPR, LWR, RBFN, BaseModelABC, LeastSquares


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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

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

    plt.tight_layout()
    plt.show()


def _model_imitate(data, x, xd, model, num_gaussians=None, starting=0, t_end=10, n=100):
    if num_gaussians is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        mvns = init_gaussians_ax(data, num_gaussians, ax=axes[0])
        ax_idx = 1
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_idx = 0

    # starting point for imitation
    x0 = data[starting][0]
    x_rk4, t_tk4 = model.imitate(x0, t_end=t_end)

    # plots for generated trajectory
    plot_curves_ax(axes[ax_idx], data, alpha=0.3, c="g", label="demonstrations")
    plot_curves_ax(
        axes[ax_idx], x_rk4[None], show_start_end=False, label="generated trajectory"
    )
    axes[ax_idx].set_title("Generated Trajectory")

    # vector field plot using stream line
    ax_idx += 1
    plot_curves_ax(axes[ax_idx], data, alpha=0.5, c="b", label="demonstrations")
    streamplot_ax(
        axes[ax_idx],
        model.predict,
        x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
        y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
        width=3,
        color="g",
        n=n,
    )
    axes[ax_idx].set_title("Vector Field")

    plt.tight_layout()
    plt.show()


def fit_lwr(dataset, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = LWR(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=6)


def fit_rbfn(dataset, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=0)


def fit_gmr(dataset, n=10):
    data, x, xd = load_data_ax(dataset)
    model = GMR(n_mixture=n)
    model.fit(x, xd, data)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=0)


def fit_gpr(dataset, kernel, alpha=1, num_traj=1, show_prior_posterior=False):
    def plot_vector_field(data, x, xd, model, n, ax=None, title=None):
        """
        n: number of trajectories selected for training
        """
        plot_curves_ax(ax, data[:n], alpha=0.5, c="b", label="demonstrations")
        streamplot_ax(
            ax,
            model.sample,
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
            n=50,
        )
        if title:
            ax.set_title(title)

    data, x, xd = load_data_ax(dataset)
    model = GPR(kernel=kernel, alpha=alpha)
    x_new, xd_new = select_trajectories(data, x, xd, num_traj)
    if show_prior_posterior:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_vector_field(data, x, xd, model, num_traj, ax=axes[0], title="Prior")
    model.fit(x_new, xd_new)
    if show_prior_posterior:
        plot_vector_field(data, x, xd, model, num_traj, ax=axes[1], title="Posterior")
    _model_imitate(data, x, xd, model, starting=num_traj - 1, t_end=5, n=100)


def select_trajectories(data, x, xd, n):
    """
    n: number of trajectories selected for training
    """
    x_new = x.reshape(*data.shape)
    x_new = x_new[:n].reshape(-1, 2)
    xd_new = xd.reshape(*data.shape)
    xd_new = xd_new[:n].reshape(-1, 2)
    return x_new, xd_new


def _get_model(model_key, model_params):
    if model_key == "lwr":
        model = LWR(mvns=model_params["mvns"], bias=model_params["bias"])
    elif model_key == "rbfn":
        model = RBFN(mvns=model_params["mvns"], bias=model_params["bias"])
    elif model_key == "gmr":
        n = model_params.get("n", model_params.get("n_mixture"))
        model = GMR(n_mixture=n)
    else:
        raise ValueError("Unsupported model")
    return model


def _different_initial_points(dataset, model: BaseModelABC, initial_points):
    """
    Generates stacked plots for different starting points
    """
    data, x, xd = load_data_ax(dataset)

    k = len(initial_points)
    fig, axes = plt.subplots(k, 2, figsize=(12, 5 * k))

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
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = LWR(mvns, bias=bias)
    model.fit(x, xd)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_rbfn(dataset, initial_points, n=4, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_gmr(dataset, initial_points, n=10):
    data, x, xd = load_data_ax(dataset)
    model = GMR(n_mixture=n)
    model.fit(x, xd, data)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def _generalisation(dataset, model_key, model_params, n_values):
    """
    Generates len(n_values)x2 plots for different numbers of Gaussians.

    Params:
        dataset: dataset to use
        model_key: model to use
        model_params: additional parameters for model
        n_values: list of number of Gaussians to use
    """
    data, x, xd = load_data_ax(dataset)
    k = len(n_values)
    fig, axes = plt.subplots(k, 3, figsize=(12, 4 * k))

    for i, n in enumerate(n_values):
        mvns = init_gaussians_ax(data, n, ax=axes[i, 0])
        axes[i, 0].set_title(f"Data for {n} Gaussians")

        model_params = {"mvns": mvns, "n": n, **model_params}
        model = _get_model(model_key, model_params)
        if model_key == "gmr":
            model.fit(x, xd, data)
        else:
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
        model_key="lwr",
        n_values=n_values,
        model_params={"bias": bias},
    )


def generalisation_rbfn(dataset, n_values, bias=False):
    _generalisation(
        dataset=dataset,
        model_key="rbfn",
        n_values=n_values,
        model_params={"bias": bias},
    )


def generalisation_gmr(dataset, n_values):
    _generalisation(
        dataset=dataset,
        model_key="gmr",
        n_values=n_values,
        model_params={},
    )
