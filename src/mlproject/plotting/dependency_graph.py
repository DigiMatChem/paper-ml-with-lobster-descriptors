import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_dependency_graph_from_df(
    results_df: pd.DataFrame,
    feature1_name: str = "Lobster Features",
    feature2_name: str = "Matminer Features",
    target_name: str = "Target",
    metric: str = "MAE Mean",
    figsize: tuple = (10, 10),
    node_colors: dict | None = None,
    title: str = "",
    save_path: str | None = None,
) -> None:
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results_df.")

    def get_metric_value(frm, to):
        row = results_df[(results_df["From"] == frm) & (results_df["To"] == to)]
        if row.empty:
            return np.nan
        return row[metric].values[0]

    r_feature1_target = get_metric_value(feature1_name, target_name)
    r_feature2_target = get_metric_value(feature2_name, target_name)
    r_feature1_feature2 = get_metric_value(feature1_name, feature2_name)
    r_feature2_feature1 = get_metric_value(feature2_name, feature1_name)

    default_colors = {
        feature1_name: "#7fc7ff",  # blue
        feature2_name: "#a5d6a7",  # green
        target_name: "#ffcc80",  # orange
    }
    colors = {**default_colors, **(node_colors or {})}

    G = nx.MultiDiGraph()
    G.add_nodes_from([feature1_name, feature2_name, target_name])

    G.add_edge(
        feature1_name,
        target_name,
        key="f1_target",
        value=r_feature1_target,
        color=colors[feature1_name],
    )
    G.add_edge(
        feature2_name,
        target_name,
        key="f2_target",
        value=r_feature2_target,
        color=colors[feature2_name],
    )
    G.add_edge(
        feature1_name,
        feature2_name,
        key="f1_f2",
        value=r_feature1_feature2,
        connectionstyle="arc3,rad=0.2",
        color=colors[feature1_name],
    )
    G.add_edge(
        feature2_name,
        feature1_name,
        key="f2_f1",
        value=r_feature2_feature1,
        connectionstyle="arc3,rad=0.2",
        color=colors[feature2_name],
    )

    pos = {
        feature1_name: (0.3, 0.3),
        feature2_name: (0.7, 0.3),
        target_name: (0.5, 0.5),
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Draw nodes
    for node, color in colors.items():
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_color=color, node_size=2000, ax=ax
        )

    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Draw edges
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_color = data.get("color", "k")
        if "connectionstyle" in data:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                connectionstyle=data["connectionstyle"],
                edge_color=[edge_color],  # list for consistency
                style="dashed",
                arrows=True,
                arrowsize=25,
                width=1.5,
                ax=ax,
                min_source_margin=20,
                min_target_margin=20,
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                edge_color=[edge_color],
                style="dashed",
                arrows=True,
                arrowsize=25,
                width=1.5,
                ax=ax,
                min_source_margin=20,
                min_target_margin=20,
            )

    # Annotate edges with matching color
    for u, v, key, data in G.edges(keys=True, data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        offset = (
            -0.03
            if (u, v) == (feature1_name, feature2_name)
            else (0.03 if (u, v) == (feature2_name, feature1_name) else 0)
        )

        ax.text(
            xm,
            ym + offset,
            f"{metric}={data['value']:.2f}",
            fontsize=12,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none"),
            color=data.get("color", "k"),
        )

    ax.set_axis_off()
    # Custom title placement
    fig.text(0.5, 0.55, title, ha="center", va="center", fontsize=12, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=300, pad_inches=0, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_learnability(
    results: pd.DataFrame,
    title: str = "Feature Learnability",
    n_feats: int = 20,
    save_path: str | None = None,
) -> None:
    """
    Create a horizontal bar chart visualization of R² (mean ± std).

    Parameters
    ----------
    results : pd.DataFrame
        Must contain 'R2_mean' and 'R2_std'.
    title : str
        Plot title.
    n_feats: int
        Number of top learned features to plot
    save_path : str, optional
        If provided, saves the figure.
    """

    required_cols = {"R2_mean", "R2_std"}
    if not required_cols.issubset(results.columns):
        raise ValueError(f"Input results must contain columns: {required_cols}")

    # Take top 20 features by R²
    df = results.copy()
    df = df.sort_values("R2_mean", ascending=True).tail(n_feats)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.barh(
        df.index.astype(str),
        df["R2_mean"],
        xerr=df["R2_std"],
        capsize=4,
        color="#a6cee3",
        ecolor="black",
        edgecolor="black",
        linewidth=0.1,
    )

    ax.set_xlabel("R² (mean ± std)", fontsize=14)
    ax.set_title(title, size=14, pad=12, fontsize=18)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()
