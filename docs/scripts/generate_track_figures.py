"""Figures for the three tutorial tracks.

Each panel is the function written below. These are illustrations for the
track cards, in the same style as the tutorial figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "docs" / "_static"

TEAL = "#237f75"
ORANGE = "#c17c30"
INK = "#1f2937"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": "#334155",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
    }
)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e8eb", linewidth=0.6, zorder=0)
    ax.tick_params(length=3, width=0.7)
    ax.set_axisbelow(True)


def finish(fig, name):
    fig.savefig(STATIC / name, dpi=160, facecolor="white")
    plt.close(fig)


def nonlinear_mean():
    """Points around a sine-shaped mean and the straight line through them."""
    rng = np.random.default_rng(4)
    x = np.linspace(0, 1, 70)
    mean = np.sin(2 * np.pi * x)
    y = mean + rng.normal(0, 0.18, size=x.shape)
    line = np.polyval(np.polyfit(x, y, 1), x)
    point = "#6b7280"
    fig, ax = plt.subplots(figsize=(5.6, 2.85))
    ax.scatter(x, y, s=18, color=point, alpha=0.75, linewidths=0, label="Observations", zorder=2)
    ax.plot(x, line, color=ORANGE, lw=2.0, label="Linear fit", zorder=3)
    ax.plot(x, mean, color=TEAL, lw=2.2, label="Mean", zorder=4)
    ax.set(xlabel="x", ylabel="y")
    ax.legend(frameon=False, handlelength=1.8, borderaxespad=0.2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.7, zorder=0)
    ax.tick_params(length=3.5, width=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.18)
    finish(fig, "track_intro.png")


def spherical_surface():
    """Equirectangular display of the latitude-longitude surface used for spherical smooths."""
    longitude = np.linspace(-180, 180, 241)
    latitude = np.linspace(-90, 90, 121)
    grid_longitude, grid_latitude = np.meshgrid(longitude, latitude)
    latitude_rad = np.deg2rad(grid_latitude)
    longitude_rad = np.deg2rad(grid_longitude)
    surface = np.cos(latitude_rad) * np.sin(longitude_rad) + 0.4 * np.sin(latitude_rad)
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    image = ax.pcolormesh(
        grid_longitude,
        grid_latitude,
        surface,
        shading="auto",
        cmap="coolwarm",
    )
    ax.contour(
        grid_longitude,
        grid_latitude,
        surface,
        levels=8,
        colors="#1f2937",
        linewidths=0.45,
        alpha=0.45,
    )
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Surface on the sphere")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("auto")
    ax.tick_params(length=3, width=0.7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Surface")
    colorbar.outline.set_linewidth(0.6)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.18)
    finish(fig, "track_statistical.png")


def interaction_surface():
    """Interaction heatmap and slices, the picture used for joint feature effects."""
    values = np.linspace(-1, 1, 90)
    grid_x, grid_z = np.meshgrid(values, values)
    surface = 1.6 * grid_x * grid_z + 0.85 * np.exp(
        -7 * ((grid_x - 0.35) ** 2 + (grid_z + 0.25) ** 2)
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.15))
    image = axes[0].pcolormesh(grid_x, grid_z, surface, cmap="coolwarm", shading="auto")
    axes[0].set(xlabel="x", ylabel="z", title="Interaction")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.outline.set_linewidth(0.6)
    for index, color in zip((12, 45, 74), ("#3b6ea5", ORANGE, TEAL), strict=True):
        axes[1].plot(values, surface[index], color=color, lw=2.0, label=f"z={values[index]:.2f}")
    axes[1].set(xlabel="x", ylabel="Contribution", title="Slices at fixed z")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        style(ax)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.18, wspace=0.42)
    finish(fig, "track_neural.png")


if __name__ == "__main__":
    nonlinear_mean()
    spherical_surface()
    interaction_surface()
