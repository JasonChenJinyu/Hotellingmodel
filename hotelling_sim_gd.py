#!/usr/bin/env python3
"""Hotelling model (gradient-ascent best responses with smooth demand).

Each round, every firm solves a (smooth) profit maximization problem over its
location along [0, 1], holding rivals fixed at their previous positions. Profit
uses a soft-assignment of customers based on distance (logit/softmax over
negative distance) so the objective is differentiable. We then update all firms
simultaneously to their (approximate) argmax found via gradient ascent.

GUI provides sliders to tune learning rate, iterations per round, demand
"temperature" beta, and grid resolution. Also supports changing firm count and
initial positions live.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, CheckButtons, Button, TextBox


@dataclass
class Config:
    num_firms: int
    positions: np.ndarray
    frames: int
    loop: bool


def default_config() -> Config:
    n = 3
    pos = np.linspace(0.1, 0.9, n)
    frames = 10**9  # effectively unlimited; time axis auto-scrolls
    loop = True
    return Config(n, pos, frames, loop)


def compute_segments(positions: np.ndarray) -> List[tuple[int, float, float]]:
    order = np.argsort(positions)
    xp = positions[order]
    segs: List[tuple[int, float, float]] = []
    for k, firm_idx in enumerate(order):
        left = 0.0 if k == 0 else 0.5 * (xp[k - 1] + xp[k])
        right = 1.0 if k == len(xp) - 1 else 0.5 * (xp[k] + xp[k + 1])
        segs.append((int(firm_idx), left, right))
    return segs


def soft_shares(positions: np.ndarray, beta: float, grid_n: int) -> np.ndarray:
    """Approximate market shares with a softmax over negative distances.

    Share_i = integral_s softmax_i(-beta * |s - x_i|) ds over s in [0,1].
    We approximate the integral with a uniform grid of points.
    """
    n = len(positions)
    s = np.linspace(0.0, 1.0, grid_n)
    # distances: [n, grid_n]
    d = np.abs(positions[:, None] - s[None, :])
    # logits = -beta * distance
    logits = -beta * d
    # subtract max per column for numerical stability
    logits = logits - logits.max(axis=0, keepdims=True)
    w = np.exp(logits)
    probs = w / (w.sum(axis=0, keepdims=True) + 1e-12)
    share = probs.mean(axis=1)  # average over s ~ Uniform[0,1]
    return share


def firm_profit(i: int, x_i: float, positions: np.ndarray, beta: float, grid_n: int) -> float:
    pos = positions.copy()
    pos[i] = float(np.clip(x_i, 0.0, 1.0))
    return float(soft_shares(pos, beta, grid_n)[i])


def grad_profit_numeric(i: int, x_i: float, positions: np.ndarray, beta: float, grid_n: int, h: float = 1e-3) -> float:
    h = max(1e-6, min(1e-1, float(h)))
    a = firm_profit(i, x_i - h, positions, beta, grid_n)
    b = firm_profit(i, x_i + h, positions, beta, grid_n)
    return (b - a) / (2.0 * h)


def argmax_by_gradient(
    i: int,
    positions: np.ndarray,
    beta: float,
    grid_n: int,
    steps: int,
    lr: float,
    h: float,
    momentum: float,
    tol: float,
    backtrack: bool,
) -> float:
    x = float(np.clip(positions[i], 0.0, 1.0))
    v = 0.0
    small = float(max(1e-8, tol))
    stuck = 0
    for _ in range(max(1, steps)):
        g = grad_profit_numeric(i, x, positions, beta, grid_n, h=h)
        if abs(g) < small:
            stuck += 1
            if stuck >= 3:
                break
        else:
            stuck = 0
        v = momentum * v + (1.0 - momentum) * g
        step = lr * v
        if backtrack:
            cur = firm_profit(i, x, positions, beta, grid_n)
            t = 1.0
            accepted = False
            for _k in range(6):
                xn = float(np.clip(x + t * step, 0.0, 1.0))
                nxt = firm_profit(i, xn, positions, beta, grid_n)
                if nxt >= cur:
                    x = xn
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                # if no improving step found, just do a tiny move
                x = float(np.clip(x + 0.1 * step, 0.0, 1.0))
        else:
            x = float(np.clip(x + step, 0.0, 1.0))

    # Optional refine: also consider edges
    vals = [firm_profit(i, x, positions, beta, grid_n), firm_profit(i, 0.0, positions, beta, grid_n), firm_profit(i, 1.0, positions, beta, grid_n)]
    cand = [x, 0.0, 1.0]
    return float(cand[int(np.argmax(vals))])


def simultaneous_gradient_step(
    positions: np.ndarray,
    beta: float,
    grid_n: int,
    steps: int,
    lr: float,
    h: float,
    momentum: float,
    tol: float,
    backtrack: bool,
) -> np.ndarray:
    n = len(positions)
    pos = positions.copy()
    new_pos = pos.copy()
    for i in range(n):
        new_pos[i] = argmax_by_gradient(i, pos, beta, grid_n, steps, lr, h, momentum, tol, backtrack)
    return new_pos


def profit_curves_for_all(
    positions: np.ndarray,
    beta: float,
    grid_n: int,
    curve_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute profit curves p_i(x) for all firms at current rivals' positions.

    Returns xs (K,) and profits (n, K). Uses numerically stable vectorization.
    """
    n = len(positions)
    K = max(30, int(curve_samples))
    xs = np.linspace(0.0, 1.0, K)
    S = max(50, int(grid_n))
    s = np.linspace(0.0, 1.0, S)
    profits = np.empty((n, K), dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        others = positions[mask][:, None]  # [m,1]
        logits_others = -beta * np.abs(others - s[None, :])  # [m,S]
        m_others = logits_others.max(axis=0)  # [S]
        sum_others_e = np.exp(logits_others - m_others[None, :]).sum(axis=0)  # [S]

        logits_i = -beta * np.abs(xs[:, None] - s[None, :])  # [K,S]
        v = logits_i - m_others[None, :]  # [K,S]
        ev = np.exp(v)
        prob = np.where(
            v <= 0,
            ev / (ev + sum_others_e[None, :]),
            1.0 / (1.0 + sum_others_e[None, :] * np.exp(-v)),
        )
        profits[i, :] = prob.mean(axis=1)

    return xs, profits


def build_animation(config: Config) -> None:
    positions = config.positions.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, config.num_firms))

    fig = plt.figure(figsize=(13, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[2.2, 2.0], hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.35, 0.55)
    ax.set_xlabel("Consumer location (0 → 1)")
    ax.set_yticks([])
    ax.set_title("Hotelling dynamics (gradient-ascent best responses)")
    ax.hlines(0, 0, 1, colors="gray", linewidth=2)

    scatter = ax.scatter(positions, np.zeros_like(positions), s=160, c=colors, zorder=3)
    labels = [
        ax.text(x, 0.08, f"Firm {idx + 1}", ha="center", color=colors[idx], fontsize=9, weight="bold")
        for idx, x in enumerate(positions)
    ]

    share_text = ax.text(
        1.02,
        0.5,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        family="monospace",
    )

    share_patches: List[Rectangle] = []
    share_labels = []
    for idx in range(config.num_firms):
        patch = Rectangle((0, -0.28 + 0.05 * idx), 0.0, 0.035, color=colors[idx], alpha=0.35)
        ax.add_patch(patch)
        share_patches.append(patch)
        share_labels.append(
            ax.text(0, -0.265 + 0.05 * idx, "", color=colors[idx], fontsize=8, ha="left")
        )

    history_window = 1000
    ax_ts.set_xlim(0, history_window)
    ax_ts.set_ylim(0, 1)
    ax_ts.set_xlabel("Time (frame)")
    ax_ts.set_ylabel("Position")
    ax_ts.set_title("Positions over time")

    ts_lines = []
    ts_x = [0]
    ts_y: List[List[float]] = [[float(positions[i])] for i in range(config.num_firms)]
    for i in range(config.num_firms):
        (line,) = ax_ts.plot(ts_x, ts_y[i], color=colors[i], lw=1.8, label=f"Firm {i+1}")
        ts_lines.append(line)
    if config.num_firms <= 10:
        ax_ts.legend(loc="upper right", fontsize=8, frameon=False)

    prev_order = np.argsort(positions)
    swap_artists: List[plt.Artist] = []
    # Profit overlay state
    y_base, y_scale = 0.15, 0.35  # map profit [0,1] -> y in [0.15, 0.50]
    curve_lines: List[plt.Line2D] = []
    next_markers: List[plt.Artist] = []

    # Controls
    ax_lr = fig.add_axes([0.09, 0.12, 0.36, 0.03])
    ax_steps = fig.add_axes([0.09, 0.08, 0.36, 0.03])
    ax_beta = fig.add_axes([0.55, 0.12, 0.16, 0.03])
    ax_grid = fig.add_axes([0.55, 0.08, 0.16, 0.03])
    ax_mom = fig.add_axes([0.09, 0.04, 0.36, 0.03])
    ax_tol = fig.add_axes([0.55, 0.04, 0.16, 0.03])
    ax_h = fig.add_axes([0.09, 0.00, 0.36, 0.03])
    ax_chk = fig.add_axes([0.75, 0.05, 0.12, 0.12])
    ax_reset = fig.add_axes([0.87, 0.085, 0.10, 0.04])
    ax_stepbtn = fig.add_axes([0.87, 0.035, 0.10, 0.04])
    ax_firms = fig.add_axes([0.09, 0.02, 0.36, 0.03])
    ax_posbox = fig.add_axes([0.55, 0.02, 0.29, 0.035])
    ax_apply = fig.add_axes([0.87, 0.005, 0.10, 0.04])
    ax_ok = fig.add_axes([0.55, 0.00, 0.16, 0.03])
    ax_or = fig.add_axes([0.75, 0.00, 0.10, 0.03])

    s_lr = Slider(ax_lr, 'LR (eta)', 1e-4, 5e-1, valinit=5e-2, valstep=1e-4)
    s_steps = Slider(ax_steps, 'Iters/round', 1, 200, valinit=30, valstep=1)
    s_beta = Slider(ax_beta, 'Beta', 1.0, 400.0, valinit=80.0, valstep=1.0)
    s_grid = Slider(ax_grid, 'Demand grid', 100, 4000, valinit=600, valstep=10)
    s_mom = Slider(ax_mom, 'Momentum', 0.0, 0.95, valinit=0.8, valstep=0.01)
    s_tol_log = Slider(ax_tol, 'log10 |g| tol', -6.0, -2.0, valinit=-4.0, valstep=0.1)
    s_h_log = Slider(ax_h, 'log10 h (grad)', -5.0, -2.0, valinit=-3.0, valstep=0.1)
    cbtn = CheckButtons(ax_chk, ['Loop', 'Pause', 'Backtrack', 'Overlay', 'Swaps'], [config.loop, False, True, True, False])
    b_reset = Button(ax_reset, 'Reset pos')
    b_step = Button(ax_stepbtn, 'Step once')
    s_firms = Slider(ax_firms, 'Firms', 2, 12, valinit=config.num_firms, valstep=1)
    t_pos = TextBox(ax_posbox, 'Init pos: ', initial=" ".join(f"{p:.2f}" for p in positions))
    b_apply = Button(ax_apply, 'Apply')
    s_ok = Slider(ax_ok, 'Overlay K', 20, 200, valinit=80, valstep=5)
    s_or = Slider(ax_or, 'Overlay every', 1, 20, valinit=5, valstep=1)

    paused = False
    loop_flag = config.loop
    use_backtrack = True
    show_overlay = True
    show_swaps = False

    def on_check(label):
        nonlocal paused, loop_flag, use_backtrack, show_overlay, show_swaps, swap_artists
        if label == 'Pause':
            paused = not paused
        elif label == 'Loop':
            loop_flag = not loop_flag
        elif label == 'Backtrack':
            use_backtrack = not use_backtrack
        elif label == 'Overlay':
            show_overlay = not show_overlay
            if not show_overlay:
                # hide overlay artists
                for ln in curve_lines:
                    ln.set_visible(False)
                for mk in next_markers:
                    mk.set_visible(False)
                fig.canvas.draw_idle()
            else:
                for ln in curve_lines:
                    ln.set_visible(True)
                for mk in next_markers:
                    mk.set_visible(True)
                update_profit_overlay()
        elif label == 'Swaps':
            show_swaps = not show_swaps
            if not show_swaps:
                for art in swap_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                swap_artists.clear()
                fig.canvas.draw_idle()
    cbtn.on_clicked(on_check)

    init_positions = positions.copy()

    def update_profit_overlay():
        nonlocal curve_lines, next_markers
        if not show_overlay:
            return
        beta = float(s_beta.val)
        grid_n = min(int(s_grid.val), 400)  # cap overlay integration grid for speed
        steps = int(s_steps.val)
        lr = float(s_lr.val)
        tol = 10 ** float(s_tol_log.val)
        h = 10 ** float(s_h_log.val)
        momentum = float(s_mom.val)

        # Curves (samples controlled by slider; use capped grid for speed)
        xs, profits = profit_curves_for_all(positions, beta, grid_n, curve_samples=int(s_ok.val))
        ymap = y_base + y_scale * profits
        # Build or update lines
        if len(curve_lines) != config.num_firms:
            for ln in curve_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            curve_lines.clear()
            for i in range(config.num_firms):
                (ln,) = ax.plot(xs, ymap[i], color=colors[i], alpha=0.6, lw=1.2)
                curve_lines.append(ln)
        else:
            for i in range(config.num_firms):
                curve_lines[i].set_data(xs, ymap[i])

        # Next-step markers (predicted argmax from profit curves for speed)
        proposed = np.array([xs[int(np.argmax(profits[i]))] for i in range(config.num_firms)], dtype=float)
        if len(next_markers) != config.num_firms:
            for art in next_markers:
                try:
                    art.remove()
                except Exception:
                    pass
            next_markers.clear()
            for i in range(config.num_firms):
                yv = y_base + y_scale * profits[i, int(np.argmax(profits[i]))]
                sc = ax.scatter([proposed[i]], [yv], s=36, c=[colors[i]], marker='^', zorder=5, edgecolor='k', linewidths=0.3)
                next_markers.append(sc)
        else:
            for i, sc in enumerate(next_markers):
                yv = y_base + y_scale * profits[i, int(np.argmax(profits[i]))]
                sc.set_offsets(np.array([[proposed[i], yv]]))

    def do_reset(event):
        nonlocal positions, ts_x, ts_y, prev_order, swap_artists
        positions = init_positions.copy()
        ts_x[:] = [0]
        for i in range(config.num_firms):
            ts_y[i][:] = [float(positions[i])]
            ts_lines[i].set_data(ts_x, ts_y[i])
        prev_order = np.argsort(positions)
        for art in swap_artists:
            try:
                art.remove()
            except Exception:
                pass
        swap_artists.clear()
        fig.canvas.draw_idle()
    b_reset.on_clicked(do_reset)
    # Initialize overlay
    update_profit_overlay()

    def parse_positions(text: str, n: int) -> np.ndarray:
        cleaned = text.replace(',', ' ').strip()
        if not cleaned:
            return np.linspace(0.1, 0.9, n)
        parts = cleaned.split()
        vals = []
        for p in parts:
            try:
                v = float(p)
                if 0.0 <= v <= 1.0:
                    vals.append(v)
            except Exception:
                pass
        if len(vals) != n:
            return np.linspace(0.1, 0.9, n)
        return np.array(vals, dtype=float)

    def reinit(new_positions: np.ndarray):
        nonlocal positions, colors, scatter, labels, share_patches, share_labels, ts_lines, ts_x, ts_y, prev_order, swap_artists, init_positions
        try:
            scatter.remove()
        except Exception:
            pass
        for artist_list in (labels, share_patches, share_labels, ts_lines, swap_artists):
            for art in artist_list:
                try:
                    art.remove()
                except Exception:
                    pass
        labels.clear(); share_patches.clear(); share_labels.clear(); ts_lines.clear(); swap_artists.clear()

        config.num_firms = int(len(new_positions))
        positions = np.array(new_positions, dtype=float)
        colors = plt.cm.tab10(np.linspace(0, 1, config.num_firms))

        scatter = ax.scatter(positions, np.zeros_like(positions), s=160, c=colors, zorder=3)
        for idx, x in enumerate(positions):
            labels.append(ax.text(x, 0.08, f"Firm {idx + 1}", ha='center', color=colors[idx], fontsize=9, weight='bold'))
        for idx in range(config.num_firms):
            patch = Rectangle((0, -0.28 + 0.05 * idx), 0.0, 0.035, color=colors[idx], alpha=0.35)
            ax.add_patch(patch)
            share_patches.append(patch)
            share_labels.append(ax.text(0, -0.265 + 0.05 * idx, "", color=colors[idx], fontsize=8, ha='left'))

        ts_x[:] = [0]
        ts_y[:] = []
        for i in range(config.num_firms):
            ts_y.append([float(positions[i])])
            (line,) = ax_ts.plot(ts_x, ts_y[i], color=colors[i], lw=1.8, label=f"Firm {i+1}")
            ts_lines.append(line)
        if config.num_firms <= 10:
            ax_ts.legend(loc='upper right', fontsize=8, frameon=False)

        prev_order = np.argsort(positions)
        init_positions = positions.copy()
        s_firms.set_val(config.num_firms)
        t_pos.set_val(" ".join(f"{p:.2f}" for p in positions))
        fig.canvas.draw_idle()
        update_profit_overlay()

    def on_apply(event):
        n = int(s_firms.val)
        new_pos = parse_positions(t_pos.text, n)
        reinit(new_pos)
    b_apply.on_clicked(on_apply)

    def do_step(event=None):
        nonlocal positions
        beta = float(s_beta.val)
        grid_n = int(s_grid.val)
        steps = int(s_steps.val)
        lr = float(s_lr.val)
        tol = 10 ** float(s_tol_log.val)
        h = 10 ** float(s_h_log.val)
        momentum = float(s_mom.val)
        positions[:] = simultaneous_gradient_step(positions, beta, grid_n, steps, lr, h, momentum, tol, use_backtrack)
    b_step.on_clicked(do_step)

    overlay_counter = 0

    def animate(frame: int):
        nonlocal positions, ts_x, ts_y, prev_order, swap_artists, overlay_counter
        if not paused:
            beta = float(s_beta.val)
            grid_n = int(s_grid.val)
            steps = int(s_steps.val)
            lr = float(s_lr.val)
            tol = 10 ** float(s_tol_log.val)
            h = 10 ** float(s_h_log.val)
            momentum = float(s_mom.val)
            positions[:] = simultaneous_gradient_step(positions, beta, grid_n, steps, lr, h, momentum, tol, use_backtrack)

        scatter.set_offsets(np.column_stack([positions, np.zeros_like(positions)]))
        for idx, label in enumerate(labels):
            label.set_x(positions[idx])

        segments = compute_segments(positions)
        share_lines = []
        for firm_idx, left, right in segments:
            share = max(right - left, 0.0)
            share_lines.append(f"Firm {firm_idx + 1}: {share * 100:5.1f}%")
            share_patches[firm_idx].set_x(left)
            share_patches[firm_idx].set_width(share)
            share_labels[firm_idx].set_text(f"{share * 100:4.1f}% of demand")
            share_labels[firm_idx].set_position((left, share_labels[firm_idx].get_position()[1]))

        share_text.set_text("Market share by firm (hard Voronoi shown):\n" + "\n".join(share_lines))

        t = frame + 1
        if not paused:
            ts_x.append(t)
            for i in range(config.num_firms):
                ts_y[i].append(float(positions[i]))
                # Trim history window
                if len(ts_x) > history_window:
                    ts_x = ts_x[-history_window:]
                    ts_y[i] = ts_y[i][-history_window:]
                ts_lines[i].set_data(ts_x, ts_y[i])
            # Scroll x-axis
            if t > history_window:
                ax_ts.set_xlim(t - history_window, t)

        new_order = np.argsort(positions)
        if show_swaps and (not np.array_equal(new_order, prev_order)):
            vline = ax_ts.axvline(t, color="k", alpha=0.15, lw=1)
            swap_artists.append(vline)
            changed = np.where(new_order != prev_order)[0]
            for idx in changed:
                s = ax_ts.scatter([t], [positions[idx]], s=18, c=[colors[idx]], marker='o', zorder=4, edgecolor='k', linewidths=0.3)
                swap_artists.append(s)
            prev_order = new_order
            # cap number of swap markers to avoid memory growth
            if len(swap_artists) > 300:
                # remove oldest ~50 artists
                for art in swap_artists[:50]:
                    try:
                        art.remove()
                    except Exception:
                        pass
                swap_artists = swap_artists[50:]

        # Throttle overlay updates for performance
        refresh_every = int(s_or.val)
        overlay_counter = (overlay_counter + 1) % max(1, refresh_every)
        if overlay_counter == 0 or paused:
            update_profit_overlay()

        artists = [scatter, *labels, share_text, *share_patches, *share_labels, *ts_lines]
        if show_swaps:
            artists.extend(swap_artists)
        if show_overlay:
            artists.extend(curve_lines)
            artists.extend(next_markers)
        return artists

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=itertools.count(),  # ever-running frames
        interval=60,
        blit=False,
        repeat=True,
    )
    plt.tight_layout()
    plt.show()
    return anim


def main() -> None:
    cfg = default_config()
    _anim = build_animation(cfg)


if __name__ == "__main__":
    main()
