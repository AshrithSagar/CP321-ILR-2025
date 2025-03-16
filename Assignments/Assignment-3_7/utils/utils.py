"""
utils.py
Utility functions for ease of plotting
"""

from typing import Optional

import gmr
import matplotlib.pyplot as plt
import numpy as np

from .helpers import derivative
from .lasa import load_lasa
from .models import GMR, GPR, LWR, RBFN, SEDS, TPGMM, BaseModelABC, LeastSquares, ProMP


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


def load_data_ax(letter: str, show_plot=False, ax: plt.Axes = None):
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
        plot_curves_ax(ax, x)
    data = x
    x = x.reshape(-1, 2)
    xd = xd.reshape(-1, 2)
    plt.show()
    return data, x, xd


def load_data2_ax(letter: str, show_plot=False, ax: plt.Axes = None):
    """
    gets the trajectories coresponding to the given letter

    params:
        letter: character in ["c","j","s"]

    returns:
        data: array of shape (number of trajectories,number of timesteps,2)
        x: array of shape(number of trajectories*number of timesteps,2)
        xd: array of shape(number of trajectories*number of timesteps,2)

    """
    letter2id = dict(c=2, j=6, s=24)
    assert letter.lower() in letter2id
    _, x, _, _, _, _ = load_lasa(letter2id[letter.lower()])
    xds = []
    for i in range(x.shape[0]):
        dt = 1 / (x[i].shape[0] - 1)
        xd = np.vstack((np.zeros((1, x[i].shape[1])), np.diff(x[i], axis=0) / dt))
        xds.append(xd)
    xd = np.stack(xds)
    if show_plot:
        if ax is None:
            plt.figure(figsize=(10, 5))
        plot_curves_ax(ax, x)
    data = x
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


def plot_curves3_ax(ax: plt.Axes, x, alpha=1):
    """
    plots 2d curves

    params:
        x: array of shape (number of curves,n_steps_per_curve,2)
    """
    for t in range(x.shape[0]):
        ax.scatter(x[t][0, 0], x[t][0, 1], c="k")
        ax.scatter(x[t][-1, 0], x[t][-1, 1], c="b")
        ax.plot(x[t][:, 0], x[t][:, 1], alpha=alpha)


def fit_least_squares(dataset: str, lam=1e-2, bias=False):
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


def fit_lwr(dataset: str, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = LWR(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=6)


def fit_rbfn(dataset: str, n=10, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=0)


def fit_gmr(dataset: str, n=10):
    data, x, xd = load_data_ax(dataset)
    model = GMR(n_mixture=n)
    model.fit(x, xd, data)
    _model_imitate(data, x, xd, model, num_gaussians=n, starting=0)


def fit_gpr(dataset: str, kernel, alpha=1, num_traj=1, show_prior_posterior=False):
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


def fit_seds(dataset: str, n_mixture=10):
    data, x, xd = load_data_ax(dataset)
    attractor = data[0][-1]
    model = SEDS(attractor, n_mixture=n_mixture)
    model.fit(x, xd)
    _model_imitate(data, x, xd, model, t_end=5, n=100)


def fit_promp(dataset: str, n_dims=2, nweights_per_dim=20):
    _, x, xd = load_data2_ax(dataset)
    model = ProMP(n_dims=n_dims, nweights_per_dim=nweights_per_dim)
    model.fit(x, xd)
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    sample_trajectories(model, x, axes[0], n=100)
    condition_on_starting_point(model, x, axes[1], starting_index=0)


def select_trajectories(data, x, xd, n):
    """
    n: number of trajectories selected for training
    """
    x_new = x.reshape(*data.shape)
    x_new = x_new[:n].reshape(-1, 2)
    xd_new = xd.reshape(*data.shape)
    xd_new = xd_new[:n].reshape(-1, 2)
    return x_new, xd_new


def sample_trajectories(model: ProMP, x, ax, n=100):
    """Sample different trajectories"""
    x_lim = [np.min(x[:, :, 0]) - 10, np.max(x[:, :, 0]) + 10]
    y_lim = [np.min(x[:, :, 1]) - 10, np.max(x[:, :, 1]) + 10]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    x_sample = model.sample_trajectories(n)
    plot_curves_ax(ax, x_sample, alpha=0.2)
    ax.set_title(f"Sampled trajectories ({n=})")


def condition_on_starting_point(model: ProMP, x, ax, starting_index=0):
    x_lim = [np.min(x[:, :, 0]) - 10, np.max(x[:, :, 0]) + 10]
    y_lim = [np.min(x[:, :, 1]) - 10, np.max(x[:, :, 1]) + 10]
    x0 = x[starting_index][-1]
    new_mean, new_cov = model.conditioning_on_xt(x0, 1.0, 0.0)
    x_sample = model.sample_trajectories(100, new_mean, new_cov)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plot_curves_ax(ax, x_sample, alpha=0.2)
    ax.set_title(f"Conditioned on starting point {starting_index}")


def _get_model(model_key, model_params):
    if model_key == "lwr":
        model = LWR(
            mvns=model_params["mvns"],
            bias=model_params["bias"],
        )
    elif model_key == "rbfn":
        model = RBFN(
            mvns=model_params["mvns"],
            bias=model_params["bias"],
        )
    elif model_key == "gmr":
        n = model_params.get("n", model_params.get("n_mixture"))
        model = GMR(n_mixture=n)
    elif model_key == "gpr":
        model = GPR(
            kernel=model_params["kernel"],
            alpha=model_params["alpha"],
        )
    elif model_key == "seds":
        model = SEDS(
            attractor=model_params["attractor"],
            n_mixture=model_params["n_mixture"],
        )
    elif model_key == "promp":
        model = ProMP(
            n_dims=model_params["n_dims"],
            nweights_per_dim=model_params["nweights_per_dim"],
        )
    elif model_key == "tpgmm":
        model = TPGMM(
            As=model_params["As"],
            Bs=model_params["Bs"],
            n_mixture=model_params["n_mixture"],
        )
    else:
        raise ValueError("Unsupported model")
    return model


def _different_initial_points(
    dataset: str, model: BaseModelABC, initial_points, streamplot_method="predict"
):
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
            getattr(model, streamplot_method),
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
        )
        axes[i, 1].set_title(f"Vector field for starting point {start_idx}")

    plt.tight_layout()
    plt.show()


def different_initial_points_lwr(dataset: str, initial_points, n=4, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = LWR(mvns, bias=bias)
    model.fit(x, xd)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_rbfn(dataset: str, initial_points, n=4, bias=False):
    data, x, xd = load_data_ax(dataset)
    mvns = init_gaussians_ax(data, n)
    model = RBFN(mvns, bias=bias)
    model.fit(x, xd)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_gmr(dataset: str, initial_points, n=10):
    data, x, xd = load_data_ax(dataset)
    model = GMR(n_mixture=n)
    model.fit(x, xd, data)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_gpr(
    dataset: str, kernel, initial_points, alpha=1, num_traj=1
):
    data, x, xd = load_data_ax(dataset)
    model = GPR(kernel=kernel, alpha=alpha)
    x_new, xd_new = select_trajectories(data, x, xd, num_traj)
    model.fit(x_new, xd_new)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_seds(dataset: str, initial_points, n_mixture=10):
    data, x, xd = load_data_ax(dataset)
    attractor = data[0][-1]
    model = SEDS(attractor, n_mixture=n_mixture)
    model.fit(x, xd)
    _different_initial_points(
        dataset=dataset, model=model, initial_points=initial_points
    )


def different_initial_points_promp(
    dataset: str, initial_points: list[int], n_dims=2, nweights_per_dim=20
):
    _, x, xd = load_data2_ax(dataset)
    model = ProMP(n_dims=n_dims, nweights_per_dim=nweights_per_dim)
    model.fit(x, xd)

    k = len(initial_points)
    _, axes = plt.subplots(k, 2, figsize=(12, 5 * k))

    for i, start_idx in enumerate(initial_points):
        sample_trajectories(model, x, axes[i, 0], n=100)
        condition_on_starting_point(model, x, axes[i, 1], starting_index=start_idx)
    plt.tight_layout()
    plt.show()


def different_initial_points_tpgmm(
    Data: np.ndarray,
    As: np.ndarray,
    Bs: np.ndarray,
    n_mixture: int,
    offsets: list[int],
):
    mp = TPGMM(As, Bs, n_mixture)
    mp.fit(Data)

    k = len(offsets)
    _, axes = plt.subplots(k, 4, figsize=(12, 3 * k))

    for i, offset in enumerate(offsets):
        # using old params of trajectory 1
        A_new, B_new = As[:, 0], Bs[:, 0]
        mp.plot_gaussians_wrt_frames_ax(Data, A_new, B_new, axes[i, :2])

        # translating start and end of the trajectory - 2  by offset
        A_new, B_new = As[:, 1].copy(), Bs[:, 1].copy()
        B_new[0] = B_new[0] + np.array([0, offset, 0])
        B_new[1] = B_new[1] + np.array([0, offset, 0])
        mp.plot_gaussians_wrt_frames_ax(Data, A_new, B_new, axes[i, 2:])

        axes[i, 0].set_ylabel(f"{offset=}")
    plt.tight_layout()
    plt.show()


def _generalisation_n(
    dataset: str,
    model_key: str,
    model_params: dict,
    n_values: list[int],
    starting: int = 0,
    t_end: int = 10,
    streamplot_method: str = "predict",
):
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
    axes = np.atleast_2d(axes)

    for i, n in enumerate(n_values):
        mvns = init_gaussians_ax(data, n, ax=axes[i, 0])
        axes[i, 0].set_title(f"Data for {n} Gaussians")

        model_params = {"mvns": mvns, "n": n, "n_mixture": n, **model_params}
        model = _get_model(model_key, model_params)
        if model_key == "gmr":
            model.fit(x, xd, data)
        else:
            model.fit(x, xd)

        x0 = data[starting][0]
        x_rk4, _ = model.imitate(x0, t_end=t_end)

        plot_curves_ax(axes[i, 1], data, alpha=0.3, c="g", label="demonstrations")
        plot_curves_ax(
            axes[i, 1], x_rk4[None], show_start_end=False, label="generated trajectory"
        )
        axes[i, 1].set_title(f"Trajectory for {n} Gaussians")

        plot_curves_ax(axes[i, 2], data, alpha=0.5, c="b", label="demonstrations")
        streamplot_ax(
            axes[i, 2],
            getattr(model, streamplot_method),
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
        )
        axes[i, 2].set_title(f"Vector field for {n} Gaussians")

    plt.tight_layout()
    plt.show()


def _generalisation_kernel(
    dataset: str,
    model_key: str,
    model_params: dict,
    kernels: dict[str, callable],
    num_traj: Optional[int] = 1,
    starting: int = 0,
    t_end: int = 10,
    streamplot_method: str = "predict",
):
    """
    Generates len(kernels)x2 plots for different kernels.

    Params:
        dataset: dataset to use
        model_key: model to use
        model_params: additional parameters for model
        kernels: list of kernels to use
        num_traj: number of trajectories to use for fit
    """
    data, x, xd = load_data_ax(dataset)
    k = len(kernels)
    fig, axes = plt.subplots(k, 2, figsize=(12, 5 * k))
    axes = np.atleast_2d(axes)

    for i, (kernel, kernel_fn) in enumerate(kernels.items()):
        model_params = {"kernel": kernel_fn, **model_params}
        model = _get_model(model_key, model_params)
        if num_traj is not None:
            x_new, xd_new = select_trajectories(data, x, xd, n=num_traj)
            model.fit(x_new, xd_new)
        else:
            model.fit(x, xd)

        x0 = data[starting][0]
        x_rk4, _ = model.imitate(x0, t_end=t_end)

        plot_curves_ax(axes[i, 0], data, alpha=0.3, c="g", label="demonstrations")
        plot_curves_ax(
            axes[i, 0], x_rk4[None], show_start_end=False, label="generated trajectory"
        )
        axes[i, 0].set_title(f"Trajectory for {kernel} kernel")

        plot_curves_ax(axes[i, 1], data, alpha=0.5, c="b", label="demonstrations")
        streamplot_ax(
            axes[i, 1],
            getattr(model, streamplot_method),
            x_axis=(min(x[:, 0]) - 15, max(x[:, 0]) + 15),
            y_axis=(min(x[:, 1]) - 15, max(x[:, 1]) + 15),
            width=3,
            color="g",
        )
        axes[i, 1].set_title(f"Vector field for {kernel} kernel")

    plt.tight_layout()
    plt.show()


def generalisation_lwr(dataset: str, n_values, bias=False):
    _generalisation_n(
        dataset=dataset,
        model_key="lwr",
        n_values=n_values,
        model_params={"bias": bias},
    )


def generalisation_rbfn(dataset: str, n_values, bias=False):
    _generalisation_n(
        dataset=dataset,
        model_key="rbfn",
        n_values=n_values,
        model_params={"bias": bias},
    )


def generalisation_gmr(dataset: str, n_values):
    _generalisation_n(
        dataset=dataset,
        model_key="gmr",
        n_values=n_values,
        model_params={},
    )


def generalisation_gpr(
    dataset: str,
    kernels: dict[str, callable],
    alpha: int = 1,
    num_traj: int = 1,
    t_end: int = 5,
):
    _generalisation_kernel(
        dataset=dataset,
        model_key="gpr",
        kernels=kernels,
        model_params={"alpha": alpha},
        num_traj=num_traj,
        t_end=t_end,
    )


def generalisation_seds(dataset: str, n_values: list[int]):
    data, x, xd = load_data_ax(dataset)
    _generalisation_n(
        dataset=dataset,
        model_key="seds",
        n_values=n_values,
        model_params={"attractor": data[0][-1]},
    )


def generalisation_promp(dataset: str, nweights_per_dim_values: list[int], n_dims=2):
    _, x, xd = load_data2_ax(dataset)

    k = len(nweights_per_dim_values)
    _, axes = plt.subplots(k, 2, figsize=(12, 5 * k))

    for i, nweights_per_dim in enumerate(nweights_per_dim_values):
        model = ProMP(n_dims=n_dims, nweights_per_dim=nweights_per_dim)
        model.fit(x, xd)
        sample_trajectories(model, x, axes[i, 0], n=100)
        condition_on_starting_point(model, x, axes[i, 1], starting_index=0)
        axes[i, 0].set_ylabel(f"{nweights_per_dim=}")
    plt.tight_layout()
    plt.show()


def generalisation_tpgmm(
    Data: np.ndarray,
    As: np.ndarray,
    Bs: np.ndarray,
    n_mixture_values: list[int],
):
    k = len(n_mixture_values)
    _, axes = plt.subplots(k, 4, figsize=(12, 3 * k))

    for i, n_mixture in enumerate(n_mixture_values):
        print(f"{i=} {n_mixture=}")

        mp = TPGMM(As, Bs, n_mixture)
        mp.fit(Data)

        # using old params of trajectory 1
        A_new, B_new = As[:, 0], Bs[:, 0]
        mp.plot_gaussians_wrt_frames_ax(Data, A_new, B_new, axes[i, :2])

        # translating start and end of the trajectory - 2  by -10
        A_new, B_new = As[:, 1].copy(), Bs[:, 1].copy()
        B_new[0] = B_new[0] + np.array([0, -10, 0])
        B_new[1] = B_new[1] + np.array([0, -10, 0])
        mp.plot_gaussians_wrt_frames_ax(Data, A_new, B_new, axes[i, 2:])

        axes[i, 0].set_ylabel(f"{n_mixture=}")
    plt.tight_layout()
    plt.show()
