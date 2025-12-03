import numpy as np
import networkx as nx
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib.rcParams['pdf.fonttype'] = 42

def blend_colors(color1, color2, t=0.5):
    """
    Blend two colors.
    t=0 returns color1, t=1 returns color2.
    """
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    blended = (1 - t) * c1 + t * c2
    return blended


def plot_feature_graph_from_df(
    results_df,
    feature1_name="Lobster Features",
    feature2_name="Matminer Features",
    target_name="Target",
    metric="MAE Mean",
    figsize=(6, 6),
    node_colors=None,
    title="",
    save_path=None
):
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
        target_name: "#ffcc80"     # orange
    }
    colors = {**default_colors, **(node_colors or {})}

    G = nx.MultiDiGraph()
    G.add_nodes_from([feature1_name, feature2_name, target_name])

    G.add_edge(feature1_name, target_name, key='f1_target', value=r_feature1_target,
               color=colors[feature1_name])
    G.add_edge(feature2_name, target_name, key='f2_target', value=r_feature2_target,
               color=colors[feature2_name])
    G.add_edge(feature1_name, feature2_name, key='f1_f2', value=r_feature1_feature2,
               connectionstyle='arc3,rad=0.2',
               color=blend_colors(colors[feature1_name], colors[feature2_name], t=0.3))
    G.add_edge(feature2_name, feature1_name, key='f2_f1', value=r_feature2_feature1,
               connectionstyle='arc3,rad=0.2',
               color=blend_colors(colors[feature2_name], colors[feature1_name], t=0.3))

    pos = {
        feature1_name: (0, 0),
        feature2_name: (1, 0),
        target_name: (0.5, 0.8)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    for node, color in colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=1200, ax=ax)

    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

    # Draw edges
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_color = data.get("color", "k")
        if 'connectionstyle' in data:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                connectionstyle=data['connectionstyle'],
                edge_color=[edge_color],  # list for consistency
                style='dashed',
                arrows=True,
                arrowsize=15,
                ax=ax
            )
        else:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color=[edge_color],
                style='dashed',
                arrows=True,
                arrowsize=15,
                ax=ax
            )

    # Annotate edges with matching color
    for u, v, key, data in G.edges(keys=True, data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        offset = -0.12 if (u, v) == (feature1_name, feature2_name) else (0.12 if (u, v) == (feature2_name, feature1_name) else 0)

        ax.text(
            xm, ym + offset,
            f"{metric}={data['value']:.2f}",
            fontsize=10, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            color=data.get("color", "k")
        )

    ax.set_axis_off()
    ax.margins(0.2)
    plt.tight_layout(pad=3)
    plt.title(f"{title}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_feature_learnability(results, title_prefix="Feature Learnability", r2_limit=(0, 1), save_path=None):
    """
    Create a radar and bar chart visualization of R² (mean ± std) from a results DataFrame.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame indexed by feature name, containing at least columns:
        'R2_mean' and 'R2_std'.
    title_prefix : str, optional
        Prefix for chart titles.
    r2_limit : tuple, optional
        y-axis limit for radar chart (default (0, 1)).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib Figure object.
    """
    
    # ---- Validate input ----
    required_cols = {"R2_mean", "R2_std"}
    if not required_cols.issubset(results.columns):
        raise ValueError(f"Input 'res' DataFrame must contain columns: {required_cols}")

    # ---- Build compact DataFrame ----
    df = pd.DataFrame({
        "Feature": results.index.astype(str),
        "R2_mean": results["R2_mean"].values,
        "R2_std":  results["R2_std"].values
    })

    # ---- Prepare radar data ----
    labels = df["Feature"].tolist()
    values = df["R2_mean"].tolist()
    values += values[:1]  # close the circle
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # ---- Create figure ----
    fig = plt.figure(figsize=(13, 6))

    # ----- Left panel: Radar chart -----
    ax1 = plt.subplot(1, 2, 1, polar=True)
    ax1.plot(angles, values, color="teal", linewidth=2)
    ax1.fill(angles, values, color="teal", alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels([])
    ax1.set_ylim(*r2_limit)
    ax1.set_title(f"{title_prefix} (Radar: R² mean)", size=14, pad=40)

    # ----- Right panel: Bar chart -----
    df_sorted = df.sort_values("R2_mean", ascending=True)
    ax2 = plt.subplot(1, 2, 2)
    ax2.barh(df_sorted["Feature"], df_sorted["R2_mean"],
             xerr=df_sorted["R2_std"], capsize=4, color="steelblue")
    ax2.set_xlabel("R² (mean ± std)")
    ax2.set_title(f"{title_prefix} (Bar Chart: R²)", size=14, pad=40)
    ax2.grid(axis='x', linestyle='--', alpha=0.6)

    # ---- Layout tweaks ----
    plt.subplots_adjust(wspace=0.35, top=0.88, bottom=0.1)
    plt.tight_layout(pad=3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()

    return None