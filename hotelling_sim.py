#!/usr/bin/env python3
"""Dynamic Hotelling model visualizer.

Prompts the user for the number of firms and their initial locations on the
unit interval, then animates how firms relocate in response to the Hotelling
competition dynamic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, CheckButtons, Button, TextBox


@dataclass
class SimulationConfig:
    num_firms: int
    initial_positions: np.ndarray
    move_rate: float
    frames: int
    loop: bool
    probe_radius: float
    samples_per_probe: int
    gain_tolerance: float


@dataclass
class EquilibriumReport:
    converged: bool
    steps: int
    final_positions: np.ndarray
    max_change: float
    message: str


def prompt_int(message: str, minimum: int) -> int:
    while True:
        raw = input(message).strip()
        try:
            val = int(raw)
            if val < minimum:
                raise ValueError
            return val
        except ValueError:
            print(f"Please enter an integer ≥ {minimum}.")


def prompt_int_default(message: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = input(message).strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Enter an integer value.")
            continue
        if not minimum <= val <= maximum:
            print(f"Value must be between {minimum} and {maximum}.")
            continue
        return val


def prompt_positions(num_firms: int) -> np.ndarray:
    """Collects initial firm positions, defaulting to evenly spaced points."""
    while True:
        raw = input(
            "Enter the initial positions (0-1) separated by spaces.\n"
            "Press Enter for evenly spaced defaults: "
        ).strip()
        if not raw:
            return np.linspace(0.1, 0.9, num_firms)

        parts = raw.replace(",", " ").split()
        if len(parts) != num_firms:
            print(f"You must provide exactly {num_firms} numbers.")
            continue
        try:
            values = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            print("Only numeric positions are allowed.")
            continue

        if np.any((values < 0) | (values > 1)):
            print("All positions must fall within [0, 1].")
            continue
        return values


def prompt_float(message: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = input(message).strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if not minimum <= val <= maximum:
            print(f"Value must be between {minimum} and {maximum}.")
            continue
        return val


def gather_config() -> SimulationConfig:
    # Defaults requested
    DEFAULT_MOVE_RATE = 0.05
    DEFAULT_FRAMES = 500
    DEFAULT_LOOP = False
    DEFAULT_PROBE_RADIUS = 0.1
    DEFAULT_SAMPLES = 20
    DEFAULT_GAIN = 1e-3

    num_firms = prompt_int("How many firms? (≥ 2): ", minimum=2)
    positions = prompt_positions(num_firms)

    change = input("Change run parameters? (y/N): ").strip().lower() in {"y", "yes"}
    if not change:
        return SimulationConfig(
            num_firms,
            positions,
            DEFAULT_MOVE_RATE,
            DEFAULT_FRAMES,
            DEFAULT_LOOP,
            DEFAULT_PROBE_RADIUS,
            DEFAULT_SAMPLES,
            DEFAULT_GAIN,
        )

    move_rate = prompt_float(
        "Move rate per step (0.05-0.5, default 0.05): ", default=DEFAULT_MOVE_RATE, minimum=0.05, maximum=0.5
    )
    frames = int(
        prompt_float(
            "How many animation frames? (50-500, default 500): ",
            default=DEFAULT_FRAMES,
            minimum=50,
            maximum=500,
        )
    )
    loop_raw = input("Loop animation when it ends? (y/N): ").strip().lower()
    loop = loop_raw in {"y", "yes"} if loop_raw else DEFAULT_LOOP
    probe_radius = prompt_float(
        "Probe radius around firm (0.05-0.5, default 0.1): ",
        default=DEFAULT_PROBE_RADIUS,
        minimum=0.05,
        maximum=0.5,
    )
    samples_per_probe = prompt_int_default(
        "Samples per probe (3-25, default 20): ", default=DEFAULT_SAMPLES, minimum=3, maximum=25
    )
    gain_tolerance = prompt_float(
        "Min gain to move (1e-5-1e-2, default 1e-3): ",
        default=DEFAULT_GAIN,
        minimum=1e-5,
        maximum=1e-2,
    )
    return SimulationConfig(
        num_firms,
        positions,
        move_rate,
        frames,
        loop,
        probe_radius,
        samples_per_probe,
        gain_tolerance,
    )


def best_response_targets(positions: np.ndarray, eps_frac: float = 0.1) -> np.ndarray:
    """Hotelling best responses with profitable jump deviations.

    For each firm (given others fixed), we compare market share under three
    candidate actions and choose the one with highest share:
      1) Interior placement between immediate neighbors at midpoint: share_mid = (R-L)/2.
      2) Become new leftmost by moving just left of left neighbor: share_left = L.
      3) Become new rightmost by moving just right of right neighbor: share_right = 1 - R.

    For current edge firms, we also allow jumping across all others to the
    opposite edge if that yields higher share. We allow targets to cross so
    swaps are visible. Epsilon keeps targets just inside/outside neighbors.
    """
    n = len(positions)
    order = np.argsort(positions)
    x = positions[order]
    targets_sorted = np.empty_like(x)
    eps_abs = 1e-3

    for i in range(n):
        xi = x[i]
        # Immediate neighbors (if any)
        L = x[i - 1] if i > 0 else None
        R = x[i + 1] if i < n - 1 else None

        # Candidate: interior midpoint
        if (L is not None) and (R is not None):
            share_mid = 0.5 * (R - L)
            target_mid = 0.5 * (L + R)
        else:
            share_mid = -1.0
            target_mid = xi

        # Candidate: become new leftmost (move just left of immediate left neighbor)
        if L is not None:
            epsL = max(eps_frac * max(xi - L, 1e-9), eps_abs)
            target_leftmost = L - epsL
            share_left = L
        else:
            # already leftmost; best we can do on left side is stay near second firm
            if n >= 2:
                epsL = max(eps_frac * max(x[1] - xi, 1e-9), eps_abs)
                target_leftmost = x[1] - epsL
                share_left = x[1]
            else:
                target_leftmost = xi
                share_left = -1.0

        # Candidate: become new rightmost (move just right of immediate right neighbor)
        if R is not None:
            epsR = max(eps_frac * max(R - xi, 1e-9), eps_abs)
            target_rightmost = R + epsR
            share_right = 1.0 - R
        else:
            # already rightmost; best we can do on right side is stay near second-last firm
            if n >= 2:
                epsR = max(eps_frac * max(xi - x[n - 2], 1e-9), eps_abs)
                target_rightmost = x[n - 2] + epsR
                share_right = 1.0 - x[n - 2]
            else:
                target_rightmost = xi
                share_right = -1.0

        # Choose the action with highest share
        shares = [share_mid, share_left, share_right]
        targets_candidates = [target_mid, target_leftmost, target_rightmost]
        best_idx = int(np.argmax(shares))
        targets_sorted[i] = targets_candidates[best_idx]

    targets = np.empty_like(positions)
    for rank, firm_idx in enumerate(order):
        targets[firm_idx] = targets_sorted[rank]
    return targets


def compute_segments(positions: np.ndarray) -> List[Tuple[int, float, float]]:
    """Return (firm_index, left_bound, right_bound) ordered along the line."""
    order = np.argsort(positions)
    sorted_pos = positions[order]
    segments: List[Tuple[int, float, float]] = []

    for rank, firm_idx in enumerate(order):
        if rank == 0:
            left = 0.0
        else:
            left = 0.5 * (sorted_pos[rank - 1] + sorted_pos[rank])

        if rank == len(sorted_pos) - 1:
            right = 1.0
        else:
            right = 0.5 * (sorted_pos[rank] + sorted_pos[rank + 1])
        segments.append((int(firm_idx), left, right))
    return segments


def market_shares(positions: np.ndarray) -> np.ndarray:
    """Compute each firm's market share (length of its Voronoi segment)."""
    segs = compute_segments(positions)
    shares = np.zeros_like(positions)
    for idx, left, right in segs:
        shares[idx] = max(right - left, 0.0)
    return shares


def simultaneous_probe_step(
    positions: np.ndarray,
    radius: float,
    samples: int,
    gain_tol: float,
    move_rate: float,
) -> np.ndarray:
    """One simultaneous local-search step for all firms.

    For each firm i, sample positions in [x_i - radius, x_i + radius], choose the
    candidate that maximizes firm i's market share holding others fixed at current
    positions, then apply movement toward that candidate (fractional by move_rate)
    if and only if the gain exceeds gain_tol. All firms compute using the same
    current profile, and then update at once.
    """
    n = len(positions)
    pos = positions.copy()
    base = market_shares(pos)

    best_x = pos.copy()
    best_share = base.copy()

    for i in range(n):
        x = float(pos[i])
        low = max(0.0, x - radius)
        high = min(1.0, x + radius)
        if high <= low:
            continue
        candidates = np.linspace(low, high, num=max(int(samples), 3))
        candidates = np.unique(np.concatenate(([x], candidates)))

        for c in candidates:
            pos_tmp = pos.copy()
            pos_tmp[i] = float(c)
            share_c = float(market_shares(pos_tmp)[i])
            if share_c > best_share[i] + 1e-12:
                best_share[i] = share_c
                best_x[i] = float(c)

    # Apply movement simultaneously
    out = pos.copy()
    for i in range(n):
        if best_share[i] > base[i] + gain_tol:
            if move_rate >= 0.999:
                out[i] = best_x[i]
            else:
                trial = float(pos[i] + move_rate * (best_x[i] - pos[i]))
                pos_trial = pos.copy()
                pos_trial[i] = trial
                share_trial = float(market_shares(pos_trial)[i])
                out[i] = trial if share_trial > base[i] + gain_tol else best_x[i]
        else:
            out[i] = pos[i]
    return np.clip(out, 0.0, 1.0)


def build_animation(config: SimulationConfig) -> None:
    positions = config.initial_positions.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, config.num_firms))

    # Layout: two subplots side-by-side
    fig = plt.figure(figsize=(13, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[2.2, 2.0], hspace=0.35, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])

    # Left: spatial competition line
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.35, 0.55)
    ax.set_xlabel("Consumer location (0 → 1)")
    ax.set_yticks([])
    ax.set_title("Hotelling location dynamics")
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
            ax.text(
                0,
                -0.265 + 0.05 * idx,
                "",
                color=colors[idx],
                fontsize=8,
                ha="left",
            )
        )

    # Right: time-series of positions
    ax_ts.set_xlim(0, config.frames)
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

    # Track order to highlight swaps
    prev_order = np.argsort(positions)
    swap_artists: List[plt.Artist] = []

    # Controls (sliders, checkboxes, buttons)
    # Place sliders in the second row using absolute axes for fine control
    ax_move = fig.add_axes([0.09, 0.10, 0.36, 0.03])
    ax_rad = fig.add_axes([0.09, 0.06, 0.36, 0.03])
    ax_samp = fig.add_axes([0.55, 0.10, 0.16, 0.03])
    ax_gain = fig.add_axes([0.55, 0.06, 0.16, 0.03])
    ax_chk = fig.add_axes([0.75, 0.05, 0.10, 0.10])
    ax_reset = fig.add_axes([0.87, 0.085, 0.10, 0.04])
    ax_pause = fig.add_axes([0.87, 0.035, 0.10, 0.04])
    # Firms/positions controls
    ax_firms = fig.add_axes([0.09, 0.02, 0.36, 0.03])
    ax_posbox = fig.add_axes([0.55, 0.02, 0.29, 0.035])
    ax_applyfp = fig.add_axes([0.87, 0.005, 0.10, 0.04])

    s_move = Slider(ax_move, 'Move rate', 0.01, 1.0, valinit=config.move_rate, valstep=0.01)
    s_rad = Slider(ax_rad, 'Probe radius', 0.01, 0.5, valinit=config.probe_radius, valstep=0.005)
    s_samp = Slider(ax_samp, 'Samples', 3, 25, valinit=config.samples_per_probe, valstep=1)
    s_gain_log = Slider(ax_gain, 'log10 gain tol', -5.0, -2.0, valinit=float(np.log10(config.gain_tolerance)), valstep=0.1)
    cbtn = CheckButtons(ax_chk, ['Loop', 'Pause'], [config.loop, False])
    b_reset = Button(ax_reset, 'Reset pos')
    b_pause = Button(ax_pause, 'Step once')
    s_firms = Slider(ax_firms, 'Firms', 2, 10, valinit=config.num_firms, valstep=1)
    t_pos = TextBox(ax_posbox, 'Init pos: ', initial=" ".join(f"{p:.2f}" for p in positions))
    b_applyfp = Button(ax_applyfp, 'Apply')

    paused = False
    loop_flag = config.loop

    def on_check(label):
        nonlocal paused, loop_flag
        if label == 'Pause':
            paused = not paused
        elif label == 'Loop':
            loop_flag = not loop_flag
    cbtn.on_clicked(on_check)

    init_positions = positions.copy()

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

    def parse_positions_text(text: str, n: int) -> np.ndarray:
        cleaned = text.replace(',', ' ').strip()
        if not cleaned:
            return np.linspace(0.1, 0.9, n)
        parts = cleaned.split()
        vals = []
        for p in parts:
            try:
                v = float(p)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                vals.append(v)
        if len(vals) != n:
            return np.linspace(0.1, 0.9, n)
        return np.array(vals, dtype=float)

    def reinit_firms(new_positions: np.ndarray):
        nonlocal positions, colors, scatter, labels, share_patches, share_labels, ts_lines, ts_x, ts_y, prev_order, swap_artists, init_positions
        # Remove old artists
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

        # Update model state
        config.num_firms = int(len(new_positions))
        positions = np.array(new_positions, dtype=float)
        colors = plt.cm.tab10(np.linspace(0, 1, config.num_firms))

        # Recreate left plot artists
        scatter = ax.scatter(positions, np.zeros_like(positions), s=160, c=colors, zorder=3)
        for idx, x in enumerate(positions):
            labels.append(
                ax.text(x, 0.08, f"Firm {idx + 1}", ha="center", color=colors[idx], fontsize=9, weight="bold")
            )
        for idx in range(config.num_firms):
            patch = Rectangle((0, -0.28 + 0.05 * idx), 0.0, 0.035, color=colors[idx], alpha=0.35)
            ax.add_patch(patch)
            share_patches.append(patch)
            share_labels.append(
                ax.text(0, -0.265 + 0.05 * idx, "", color=colors[idx], fontsize=8, ha="left")
            )

        # Recreate right plot lines
        ts_x[:] = [0]
        ts_y[:] = []
        for i in range(config.num_firms):
            ts_y.append([float(positions[i])])
            (line,) = ax_ts.plot(ts_x, ts_y[i], color=colors[i], lw=1.8, label=f"Firm {i+1}")
            ts_lines.append(line)
        # Reset legend
        if config.num_firms <= 10:
            ax_ts.legend(loc="upper right", fontsize=8, frameon=False)

        prev_order = np.argsort(positions)
        init_positions = positions.copy()
        # Update UI fields
        s_firms.set_val(config.num_firms)
        t_pos.set_val(" ".join(f"{p:.2f}" for p in positions))
        fig.canvas.draw_idle()

    def on_apply_firms(event):
        n = int(s_firms.val)
        new_pos = parse_positions_text(t_pos.text, n)
        reinit_firms(new_pos)
    b_applyfp.on_clicked(on_apply_firms)

    def do_step(event):
        nonlocal positions
        # perform one update even if paused
        positions = simultaneous_probe_step(
            positions,
            radius=float(s_rad.val),
            samples=int(s_samp.val),
            gain_tol=10.0 ** float(s_gain_log.val),
            move_rate=float(s_move.val),
        )
    b_pause.on_clicked(do_step)

    def animate(frame: int):
        nonlocal positions, ts_x, ts_y, prev_order, swap_artists
        if not paused:
            # Simultaneous local probe: update all firms each frame with slider-controlled params
            positions = simultaneous_probe_step(
                positions,
                radius=float(s_rad.val),
                samples=int(s_samp.val),
                gain_tol=10.0 ** float(s_gain_log.val),
                move_rate=float(s_move.val),
            )
        # Left plot updates
        scatter.set_offsets(np.column_stack([positions, np.zeros_like(positions)]))
        for idx, label in enumerate(labels):
            label.set_x(positions[idx])

        segments = compute_segments(positions)
        share_lines = []
        for firm_idx, left, right in segments:
            share = max(right - left, 0.0)
            share_pct = share * 100
            share_lines.append(f"Firm {firm_idx + 1}: {share_pct:5.1f}%")
            share_patches[firm_idx].set_x(left)
            share_patches[firm_idx].set_width(share)
            share_labels[firm_idx].set_text(f"{share_pct:4.1f}% of demand")
            share_labels[firm_idx].set_position((left, share_labels[firm_idx].get_position()[1]))

        share_text.set_text("Market share by firm:\n" + "\n".join(share_lines))

        # Right plot updates
        t = frame + 1
        if not paused:
            ts_x.append(t)
            for i in range(config.num_firms):
                ts_y[i].append(float(positions[i]))
                ts_lines[i].set_data(ts_x, ts_y[i])

        # Detect and mark swaps of order
        new_order = np.argsort(positions)
        if not np.array_equal(new_order, prev_order):
            # vertical marker for a swap event
            vline = ax_ts.axvline(t, color="k", alpha=0.15, lw=1)
            swap_artists.append(vline)
            # mark each firm whose rank changed
            rank_change = np.where(new_order != prev_order)[0]
            for idx in rank_change:
                s = ax_ts.scatter([t], [positions[idx]], s=18, c=[colors[idx]], marker='o', zorder=4, edgecolor='k', linewidths=0.3)
                swap_artists.append(s)
            prev_order = new_order

        return [
            scatter,
            *labels,
            share_text,
            *share_patches,
            *share_labels,
            *ts_lines,
            *swap_artists,
        ]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=config.frames,
        interval=50,
        blit=False,
        repeat=loop_flag,
    )
    plt.tight_layout()
    plt.show()
    return anim


def analyze_equilibrium(
    num_firms: int,
    initial_positions: np.ndarray,
    tolerance: float = 1e-4,
    max_iterations: int = 2000,
) -> EquilibriumReport:
    """Check convergence of Hotelling best-response dynamics.

    Uses the same update rule as the animation (best-response with epsilon
    tie-breaking). In the standard Hotelling model:
      - n = 2: a pure-strategy NE exists at co-location (both at 0.5).
      - n ≥ 3: no pure-strategy NE; dynamics do not converge to a stable interior profile.
    """
    positions = initial_positions.astype(float, copy=True)
    history = [positions.copy()]

    for step in range(1, max_iterations + 1):
        targets = best_response_targets(positions)
        max_change = float(np.max(np.abs(targets - positions)))

        if max_change < tolerance:
            if num_firms == 2:
                return EquilibriumReport(
                    True,
                    step - 1,
                    positions.copy(),
                    max_change,
                    "Converged to the Hotelling NE: both firms co-locate at the center.",
                )
            else:
                return EquilibriumReport(
                    False,
                    step - 1,
                    positions.copy(),
                    max_change,
                    "Apparent numerical fixed point, but theory predicts no NE for n≥3;\n"
                    "treat this as a tie-breaking artifact rather than a true equilibrium.",
                )

        positions = positions + 0.5 * (targets - positions)
        for past in history[-50:]:  # check recent history for cycles
            if np.max(np.abs(past - positions)) < tolerance:
                return EquilibriumReport(
                    False,
                    step,
                    positions.copy(),
                    max_change,
                    "Detected a cycle/oscillation in best responses → no pure-strategy NE.",
                )
        history.append(positions.copy())

    return EquilibriumReport(
        False,
        max_iterations,
        positions.copy(),
        float("nan"),
        "No convergence within iteration budget. Consistent with no NE for n≥3.",
    )


def describe_equilibrium(report: EquilibriumReport) -> str:
    position_str = ", ".join(f"{pos:.3f}" for pos in report.final_positions)
    header = "Equilibrium detected" if report.converged else "No equilibrium found"
    detail = (
        f"{header} after {report.steps} iterations "
        f"(max change {report.max_change:.2e}). Locations: [{position_str}].\n"
        f"{report.message}"
    )
    return detail


def main() -> None:
    try:
        config = gather_config()
    except (KeyboardInterrupt, EOFError):
        print("\nSimulation cancelled.")
        sys.exit(0)
    report = analyze_equilibrium(config.num_firms, config.initial_positions)
    print("\n" + describe_equilibrium(report) + "\n")
    _anim = build_animation(config)  # keep reference so animation survives until after show


if __name__ == "__main__":
    main()
