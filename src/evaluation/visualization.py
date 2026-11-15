import numpy as np
import matplotlib

matplotlib.use("Agg")  # sem janela: apenas gera a figura
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle


def som_grid_shape(nboxes):
    """Mesmo cálculo do grid do SOM em HyperNNWithInitClassifier.calculate_som_init."""
    x = max(1, int(np.ceil(np.sqrt(nboxes))))
    y = max(1, int(np.ceil(nboxes / float(x))))
    return x, y


def pick_feature_pair(X, y):
    """Escolhe as duas features mais correlacionadas (em módulo) com a classe positiva."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scores = []
    for j in range(X.shape[1]):
        if X[:, j].std() > 0 and y.std() > 0:
            scores.append(abs(np.corrcoef(X[:, j], y)[0, 1]))
        else:
            scores.append(0.0)
    scores = np.nan_to_num(np.array(scores))
    top = np.argsort(scores)[::-1][:2]
    return int(top[0]), int(top[1])


def _scatter_data(ax, X, y, fi, fj):
    ax.scatter(
        X[y == 0, fi], X[y == 0, fj], s=14, c="lightgray", label="classe 0"
    )
    ax.scatter(
        X[y == 1, fi], X[y == 1, fj], s=20, c="tab:blue", label="classe 1 (alvo)"
    )


def _draw_boxes(ax, mins, lengths, fi, fj, color, linestyle="-", fill=False, label=None):
    for k in range(len(mins)):
        w = abs(float(lengths[k, fi]))
        h = abs(float(lengths[k, fj]))
        ax.add_patch(
            Rectangle(
                (float(mins[k, fi]), float(mins[k, fj])),
                w,
                h,
                fill=fill,
                facecolor=to_rgba(color, 0.15) if fill else "none",
                edgecolor=color,
                linestyle=linestyle,
                linewidth=1.5,
                label=label if k == 0 else None,
            )
        )


def _draw_som_grid(ax, centers, grid_shape, fi, fj):
    """Desenha os neurônios e as ligações entre vizinhos do mapa."""
    if grid_shape is not None:
        gx, gy = grid_shape
        if gx * gy == len(centers):
            W = centers.reshape(gx, gy, -1)
            for i in range(gx):
                ax.plot(W[i, :, fi], W[i, :, fj], "-", c="tab:red", lw=1, alpha=0.7)
            for j in range(gy):
                ax.plot(W[:, j, fi], W[:, j, fj], "-", c="tab:red", lw=1, alpha=0.7)
    ax.scatter(
        centers[:, fi],
        centers[:, fj],
        c="tab:red",
        s=45,
        edgecolor="k",
        zorder=5,
        label=f"{len(centers)} neurônios SOM",
    )


def plot_boxes_figure(
    X_train,
    y_train,
    init_mins,
    init_lengths,
    final_mins,
    final_lengths,
    strategy,
    grid_shape=None,
    qe=None,
    te=None,
    title=None,
):
    """
    Retorna uma figura matplotlib com até três painéis:
      1. mapa do SOM (só para strategy == "som");
      2. caixas iniciais (quando a estratégia gera init);
      3. caixas iniciais (tracejadas) vs caixas após o treino (preenchidas).
    Retorna None se os dados tiverem menos de 2 features.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train)
    if X.ndim != 2 or X.shape[1] < 2:
        return None

    fi, fj = pick_feature_pair(X, y)
    has_init = init_mins is not None and init_lengths is not None

    panels = []
    if strategy == "som" and has_init:
        panels.append("som")
    if has_init:
        panels.append("init")
    panels.append("final")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.2 * len(panels), 4.8), squeeze=False
    )
    axes = axes[0]

    for ax, kind in zip(axes, panels):
        _scatter_data(ax, X, y, fi, fj)

        if kind == "som":
            centers = init_mins + np.abs(init_lengths) / 2.0
            _draw_som_grid(ax, centers, grid_shape, fi, fj)
            subtitle = "Mapa do SOM (neurônios e vizinhança)"
            if qe is not None and te is not None:
                subtitle += f"\nQE={qe:.3f}  TE={te:.3f}"
        elif kind == "init":
            _draw_boxes(
                ax, init_mins, init_lengths, fi, fj, "tab:green",
                label=f"{len(init_mins)} caixas iniciais",
            )
            if strategy == "som":
                centers = init_mins + np.abs(init_lengths) / 2.0
                ax.scatter(
                    centers[:, fi], centers[:, fj], c="tab:red", s=25,
                    edgecolor="k", zorder=5, label="neurônios (centros)",
                )
                subtitle = "Caixas iniciais geradas a partir do SOM"
            else:
                subtitle = f"Caixas iniciais ({strategy})"
        else:
            if has_init:
                _draw_boxes(
                    ax, init_mins, init_lengths, fi, fj, "gray",
                    linestyle="--", label="iniciais",
                )
            _draw_boxes(
                ax, final_mins, final_lengths, fi, fj, "tab:orange",
                fill=True, label="após o treino",
            )
            subtitle = "Caixas após o treino da HyperNN"

        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel(f"feature {fi}")
        ax.legend(fontsize=7, loc="best")

    # Eixos limitados à faixa dos dados: caixas maiores que isso aparecem cortadas.
    mx = 0.1 * (X[:, fi].max() - X[:, fi].min() or 1.0)
    my = 0.1 * (X[:, fj].max() - X[:, fj].min() or 1.0)
    for ax in axes:
        ax.set_xlim(X[:, fi].min() - mx, X[:, fi].max() + mx)
        ax.set_ylim(X[:, fj].min() - my, X[:, fj].max() + my)
    axes[0].set_ylabel(f"feature {fj}")

    fig.suptitle(
        f"{title or strategy.upper()}  |  features ({fi}, {fj})", fontsize=11
    )
    fig.tight_layout()
    return fig
