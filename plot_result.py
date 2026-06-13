import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import glob
import math
import re
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator, LogFormatterMathtext, NullFormatter
from pathlib import Path
from scipy.ndimage import gaussian_filter

# --- GLOBAL CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Common styling for Bias-Variance plots
colors = {
    'ZGR': '#2ca02c',
    'ReinMax': '#ffcc00',        # Bright yellow
    'ReinMax_Unscaled': '#000000',  # Black
    'ReinMax_Scaled': '#808000',
    'STGS': "#7700ff",
    'GRMC_10': "#0044ff",
    'GRMC_100': "#10cbe0",
    'MVE': "#e71010",
    'ST': '#7f7f7f',
}

legend_order = ['MVE', 'STGS', 'GRMC_10', 'GRMC_100', 'ST', 'ZGR', 'ReinMax', 'ReinMax_Unscaled', 'ReinMax_Scaled']


def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    print(f"Warning: File not found: {path}")
    return None


# ==========================================
# SHARED YLIM HELPER
# ==========================================
def set_ylim_anchored(plt_or_ax, zgr_v, st_v, fallback_top=None):
    """
    Set y-limits so that ZGR lands at 75% and ST lands at 25% of the plot height.
    Falls back to (0, fallback_top) if the ordering is unexpected or either is missing.
    Works with both plt (stateful) and an Axes object.
    """
    set_ylim = plt_or_ax.ylim if plt_or_ax is plt else plt_or_ax.set_ylim

    if zgr_v is not None and st_v is not None:
        span = 2 * (zgr_v - st_v)
        if span > 0:
            y_min = st_v  - 0.25 * span   # ST lands at 25%
            y_max = y_min + span           # ZGR lands at 75%
            set_ylim(y_min, y_max)
            return
    # fallback
    top = fallback_top if fallback_top is not None else (zgr_v * 2 if zgr_v is not None else 1.0)
    set_ylim(0, top)


# ==========================================
# 1. GREEDY VAE RESULTS
# ==========================================
def plot_greedy_results():
    print("\n--- Plotting Greedy VAE Results ---")
    results_dir = os.path.join(script_dir, "Results", "Greedy_VAE")
    estimators = ["MVE", "gumbel_rao"]
    seeds = range(5)
    max_epoch = 500

    all_data = {est: {'elbo': [], 'temp': []} for est in estimators}
    for est in estimators:
        for seed in seeds:
            file_path = os.path.join(results_dir, f"{est}_{seed}_8x64_OHE_epoch_{max_epoch}.pkl")
            data = load_pkl(file_path)
            if data is not None:
                elbo, temp = data
                all_data[est]['elbo'].append(elbo[:max_epoch])
                all_data[est]['temp'].append(temp[:max_epoch])

    if not any(all_data[est]['elbo'] for est in estimators):
        print("No greedy data available to plot.")
        return

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_colors = {'MVE': '#e71010', 'gumbel_rao': '#0044ff'}
    labels = {'MVE': 'MVE', 'gumbel_rao': 'Gumbel-Rao'}

    # Loss Curves (ELBO)
    ax_loss = axes[0]
    for est in estimators:
        if not all_data[est]['elbo']:
            continue
        elbo_arr = np.array(all_data[est]['elbo'])
        epochs = np.arange(1, elbo_arr.shape[1] + 1)
        mean_elbo = np.mean(elbo_arr, axis=0)
        std_elbo = np.std(elbo_arr, axis=0)
        color = plot_colors.get(est, 'black')
        label = labels.get(est, est)

        ax_loss.plot(epochs, mean_elbo, label=label, color=color, linewidth=2)
        ax_loss.fill_between(epochs, mean_elbo - std_elbo, mean_elbo + std_elbo, color=color, alpha=0.2)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("ELBO")
    ax_loss.set_title("Loss Curves")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Temperature Schedule
    for idx, est in enumerate(estimators):
        ax = axes[idx + 1]
        if all_data[est]['temp']:
            temp_arr = np.array(all_data[est]['temp'])
            epochs = np.arange(1, temp_arr.shape[1] + 1)
            mean_temp = np.mean(temp_arr, axis=0)
            color = plot_colors[est]
            for seed_run in temp_arr:
                ax.plot(epochs, seed_run, color=color, alpha=0.15, linewidth=2)
            ax.plot(epochs, mean_temp, label=f'Mean {labels[est]} Temp', color=color, linewidth=2.5)
            if est == 'MVE':
                ax.set_yscale('log')
                ax.set_ylabel("Temperature (log scale)")
            else:
                ax.set_ylabel("Temperature")
            ax.set_xlabel("Epoch")
            ax.set_title(f"{labels[est]} Schedule")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(plots_dir, "greedy_vae_summary.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved greedy summary plot to {output_path}")


# ==========================================
# 2. BIAS-VARIANCE RESULTS
# ==========================================
def plot_bias_variance():
    print("\n--- Plotting Bias Variance Results ---")
    results_dir = os.path.join(script_dir, "Results", "Bias_Variance")
    file_pattern = os.path.join(results_dir, 'bias_variance_data_dim_*_seed_*.pkl')
    pkl_files = glob.glob(file_pattern)

    if not pkl_files:
        print(f"Could not find any data files matching {file_pattern}.")
        return

    linestyles = {'ReinMax_Unscaled': '--', 'ReinMax_Scaled': '--'}

    for pkl_path in pkl_files:
        filename = os.path.basename(pkl_path)
        basename = filename[:-4]
        try:
            parts = basename.split('_')
            dim, seed = int(parts[4]), int(parts[6])
            data = load_pkl(pkl_path)
            if data is None:
                continue

            plt.figure(figsize=(4, 3))
            current_estimators = [e for e in legend_order if e in data]

            for est in current_estimators:
                if est not in ["ZGR", "ST"] and 'temps' in data[est]:
                    zorder = 1
                    if est in ['STGS', 'GRMC_10', 'GRMC_100']:
                        ls, marker, ms, zorder = 'none', 'o', 1, 0
                    else:
                        ls, marker, ms = linestyles.get(est, '-'), 'none', 0

                    alpha = 0.5 if 'Unscaled' in est or 'Scaled' in est else 0.8
                    plt.plot(data[est]['bias'], data[est]['variance'],
                             color=colors.get(est, 'gray'), linestyle=ls,
                             marker=marker, markersize=ms,
                             linewidth=1.0, alpha=alpha, zorder=zorder)

            for est in ["ZGR", "ST"]:
                if est in data:
                    plt.scatter(data[est]['bias'], data[est]['variance'], s=40,
                                color=colors.get(est, 'gray'), marker='o',
                                edgecolors='white', linewidths=0.5, clip_on=False)

            # Axis limits
            bias_candidates = []
            if 'ST' in data:
                bias_candidates.append(data['ST']['bias'])
            for est in ['STGS', 'GRMC_10', 'GRMC_100']:
                if est in data and 'temps' in data[est]:
                    bias_candidates.append(data[est]['bias'][-1])
            if bias_candidates:
                plt.xlim(0, max(bias_candidates) * 1.25)
            if 'ZGR' in data:
                plt.ylim(0, data['ZGR']['variance'] * 2)

            plt.xlabel('Bias²')
            plt.ylabel('Variance')
            plt.grid(False)

            output_png = os.path.join(plots_dir, f'dim_{dim}_seed_{seed}_bias_variance.png')
            plt.tight_layout()
            plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.0)
            plt.close()
            print(f"Saved plot: {output_png}")

        except Exception as e:
            print(f"Skipping {filename} due to a parsing error: {e}")

    generate_shared_legend("shared_bias_variance_legend.png", use_lines=True)

# ==========================================
# SHARED BIAS-VARIANCE PLOT HELPER
# ==========================================
def _plot_bv_file(data, dim, seed, config_type, output_png, xlabel, ylabel):
    """
    Core bias-variance scatter/line plot used by both the directional and
    magnitude variants. Saves the figure to output_png.
    """
    linestyles = {'ReinMax_Unscaled': '--', 'ReinMax_Scaled': '--'}

    plt.figure(figsize=(4, 3))
    current_estimators = [e for e in legend_order if e in data]

    for est in current_estimators:
        if est not in ["ZGR", "ST"] and 'temps' in data[est]:
            zorder = 1
            if est in ['STGS', 'GRMC_10', 'GRMC_100']:
                ls, marker, ms, zorder = 'none', 'o', 1, 0
            else:
                ls, marker, ms = linestyles.get(est, '-'), 'none', 0

            alpha = 0.5 if 'Unscaled' in est or 'Scaled' in est else 0.8
            plt.plot(data[est]['bias'], data[est]['variance'],
                     color=colors.get(est, 'gray'), linestyle=ls,
                     marker=marker, markersize=ms,
                     linewidth=1.0, alpha=alpha, zorder=zorder)

    for est in ["ZGR", "ST"]:
        if est in data:
            plt.scatter(data[est]['bias'], data[est]['variance'], s=40,
                        color=colors.get(est, 'gray'), marker='o',
                        edgecolors='white', linewidths=0.5, clip_on=False)

    # X-axis limits — include ZGR and ST so they're always in range
    bias_candidates = []
    for est in ['ZGR', 'ST']:
        if est in data:
            bias_candidates.append(data[est]['bias'])
    for est in ['STGS', 'GRMC_10', 'GRMC_100']:
        if est in data and 'temps' in data[est]:
            bias_candidates.append(data[est]['bias'][-1])
    if bias_candidates:
        plt.xlim(0, max(bias_candidates) * 1.25)

    # Y-axis: ZGR at 75%, ST at 25%
    zgr_v = data['ZGR']['variance'] if 'ZGR' in data else None
    st_v  = data['ST']['variance']  if 'ST'  in data else None
    set_ylim_anchored(plt, zgr_v, st_v)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    print(f"Saved plot: {output_png}")




# ==========================================
# 4. REINMAX HYPERPARAMETER CONTOUR
# ==========================================
def plot_reinmax_hyperparameters():
    print("\n--- Plotting Reinmax Hyperparameter Contour ---")
    data_dir = os.path.join(script_dir, "Results", "Reinmax")

    results = {}
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return
    for pkl_file in data_path.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            est = data['config']['estimator']
            temp = data['config']['temp']
            lr = data['config']['lr']
            results.setdefault(est, {})[(temp, lr)] = data.get('final_elbo', np.nan)
        except Exception:
            continue

    all_lrs = set()
    all_temps = set()
    for est in results:
        for (t, l) in results[est]:
            all_lrs.add(l)
            all_temps.add(t)

    if not all_lrs:
        print("No hyperparameter data found.")
        return

    sorted_lrs = sorted(list(all_lrs))
    sorted_temps = sorted(list(all_temps))

    x_coords = np.array(sorted_lrs)
    y_neg = sorted([-t for t in sorted_temps])
    y_pos = sorted([t for t in sorted_temps])
    y_coords = np.array(y_neg + y_pos)

    Z = np.full((len(y_coords), len(x_coords)), np.nan)
    lr_map = {lr: i for i, lr in enumerate(sorted_lrs)}
    y_map_neg = {y: i for i, y in enumerate(y_neg)}
    y_map_pos = {y: i + len(y_neg) for i, y in enumerate(y_pos)}

    for est_name in ['opt', 'MVE']:
        if est_name in results:
            for (t, lr), elbo in results[est_name].items():
                target_y = -t
                if target_y in y_map_neg and lr in lr_map:
                    Z[y_map_neg[target_y], lr_map[lr]] = elbo

    if 'reinmax' in results:
        for (t, lr), elbo in results['reinmax'].items():
            target_y = t
            if target_y in y_map_pos and lr in lr_map:
                Z[y_map_pos[target_y], lr_map[lr]] = elbo
    
    # Apply Gaussian smoothing to the Z matrix to create a smoother contour plot
    Z_orig = Z.copy()
    Z = gaussian_filter(Z, sigma=0.6, truncate=5.0)

    with plt.rc_context({
        "font.size": 40,
        "axes.labelsize": 48,
        "axes.titlesize": 48,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "legend.fontsize": 36,
        "figure.titlesize": 48,
    }):
        fig, ax = plt.subplots(figsize=(18, 22))
        XX, YY = np.meshgrid(x_coords, y_coords)

        z_valid = Z[~np.isnan(Z)]
        if len(z_valid) == 0:
            print("Z matrix contains only NaNs.")
            return
        # Z = np.where(np.isnan(Z), np.nan, np.clip(Z, np.nanmin(Z), 120))
        Z = np.where(np.isnan(Z), np.nan, np.clip(Z, max = 140))
        Zf = Z - 95
        z_min, z_max = float(np.nanmin(Zf)), float(np.nanmax(Zf))

        levels_filled = np.geomspace(z_min, z_max, 64)
        contour_filled = ax.contourf(
            XX,
            YY,
            Zf,
            levels=levels_filled,
            norm=LogNorm(vmin=z_min, vmax=z_max),
            cmap='viridis',
            extend='both',
        )

        # start = math.ceil(z_min / contour_step) * contour_step
        # stop = math.floor(z_max / contour_step) * contour_step
        # if start <= stop:
        #     levels_lines = np.arange(start, stop + 1e-8, contour_step)
        # else:
        #     levels_lines = MaxNLocator(nbins=10).tick_values(z_min, z_max)
        # if len(levels_lines) == 0:
        levels_lines = np.array(np.arange(102, 122, 2))
        contour_lines = ax.contour(XX, YY, Z, levels=levels_lines, colors='white', alpha=0.8, linewidths=2)
        # Add automatic labels to contour lines
        ax.clabel(contour_lines, fontsize=35, colors='white', inline=True, inline_spacing=10)

        def annotate_half_minimum(z_half, y_half, y_text_offset_sign):
            if z_half.size == 0 or np.all(np.isnan(z_half)):
                return
            min_flat_index = int(np.nanargmin(z_half))
            row_idx, col_idx = np.unravel_index(min_flat_index, z_half.shape)
            x_min_point = x_coords[col_idx]
            y_min_point = y_half[row_idx]
            z_min_point = float(z_half[row_idx, col_idx])

            ax.plot(
                x_min_point,
                y_min_point,
                marker='o',
                markersize=12,
                markerfacecolor='none',
                markeredgecolor='white',
                markeredgewidth=2.5,
                linestyle='none',
                zorder=20,
            )
            ax.annotate(
                f"{z_min_point:.2f}",
                xy=(x_min_point, y_min_point),
                xytext=(10, 12 * y_text_offset_sign),
                textcoords='offset points',
                color='white',
                alpha=0.8,
                fontsize=35,
                fontfamily=plt.rcParams['font.family'][0] if isinstance(plt.rcParams['font.family'], (list, tuple)) else plt.rcParams['font.family'],
                ha='left',
                va='bottom' if y_text_offset_sign > 0 else 'top',
                zorder=21,
            )

        split_idx = len(y_neg)
        annotate_half_minimum(Z_orig[:split_idx, :], y_coords[:split_idx], -1)
        annotate_half_minimum(Z_orig[split_idx:, :], y_coords[split_idx:], 1)

        # Keep major ticks range-based; use minor ticks for every tested LR/temperature.
        ax.tick_params(axis='both', which='major', width=3.0, length=12)

        # ax.set_ylabel("Temperature", fontsize=36)
        ax.set_xlabel("Initial Learning Rate", fontsize=36)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.xaxis.set_minor_locator(FixedLocator(x_coords))
        ax.xaxis.set_minor_formatter(NullFormatter())

        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        if len(y_coords) > 1:
            y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
            y_ticks = np.linspace(y_min, y_max, 11)
            ax.set_yticks(y_ticks)
        else:
            ax.set_yticks([y_coords[0]])
        ax.yaxis.set_minor_locator(FixedLocator(y_coords))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.tick_params(axis='x', which='minor', top=False, bottom=True, direction='in', length=8, width=2.0)
        ax.tick_params(axis='y', which='major', left=False, right=False,
                   labelleft=False, labelright=False)
        ax.tick_params(axis='y', which='minor', left=False, right=False,
                   labelleft=False, labelright=False)
        for spine in ax.spines.values():
            spine.set_zorder(10)

        # ax.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.6)
        x_min, x_max = 1.3 * 1e-4, 0.45 * 1e-2
        ax.set_xlim(x_min, x_max)

        # Explicit major ticks and labels requested for the x-axis.
        major_ticks = np.array([1.3e-4, 1e-3, 4e-3], dtype=float)
        ax.set_xticks(major_ticks, minor=False)

        def major_x_fmt(x, pos):
            return f"{x:.1e}"

        def major_y_fmt(y, pos):
            return f"{abs(y):1.1f}"

        ax.xaxis.set_major_formatter(FuncFormatter(major_x_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(major_y_fmt))

        plt.tight_layout()

        # Build two half-height overlay axes from the final main-axis position so
        # they align exactly with the original axis halves.
        pos = ax.get_position()

        # Upper half (ZGR): y in [0, 1], x-axis at the middle (bottom spine).
        ax_temp_reinmax = fig.add_axes([pos.x0, pos.y0 + pos.height / 2, pos.width, pos.height / 2],
                                   facecolor='none')
        ax_temp_reinmax.set_xscale('log')
        ax_temp_reinmax.set_xlim(ax.get_xlim())
        ax_temp_reinmax.set_ylim(0.0, 1.0)
        reinmax_ticks = np.linspace(0.0, 1.0, 6)
        ax_temp_reinmax.set_yticks(reinmax_ticks)
        ax_temp_reinmax.set_yticklabels(["ZGR\nMVE" if np.isclose(tick, 0.0) else f"{tick:.1f}" for tick in reinmax_ticks])
        ax_temp_reinmax.set_ylabel("Temperature ReinMax", fontsize=35)
        ax_temp_reinmax.yaxis.set_label_position('left')
        ax_temp_reinmax.yaxis.tick_left()
        ax_temp_reinmax.set_xticks(ax.get_xticks(minor=False), minor=False)
        ax_temp_reinmax.xaxis.set_minor_locator(FixedLocator(x_coords))
        ax_temp_reinmax.xaxis.set_minor_formatter(NullFormatter())
        ax_temp_reinmax.tick_params(axis='y', which='major', left=True, right=False,
                length=12, width=3.0)
        ax_temp_reinmax.tick_params(axis='y', which='minor', left=True, right=False,
                length=8, width=2.0)
        ax_temp_reinmax.tick_params(axis='x', which='major', bottom=True, top=False,
                    labelbottom=False, labeltop=False,
                    direction='inout', length=12, width=3.0)
        ax_temp_reinmax.tick_params(axis='x', which='minor', bottom=True, top=False,
                    labelbottom=False, labeltop=False,
                    direction='inout', length=8, width=2.0)
        ax_temp_reinmax.spines['bottom'].set_visible(True)
        ax_temp_reinmax.spines['top'].set_visible(False)
        ax_temp_reinmax.spines['left'].set_visible(False)
        ax_temp_reinmax.spines['right'].set_visible(False)

        # Lower half (MVE): exact bottom half, with y oriented downward from 0 at
        # the center split line to 1 at the bottom.
        ax_temp_mve = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height / 2],
                                   facecolor='none')
        ax_temp_mve.set_xscale('log')
        ax_temp_mve.set_xlim(ax.get_xlim())
        ax_temp_mve.set_ylim(1.0, 0.0)
        mve_ticks = np.linspace(0.0, 1.0, 6)
        ax_temp_mve.set_yticks(mve_ticks)
        ax_temp_mve.set_yticklabels(["ZGR\nMVE" if np.isclose(tick, 0.0) else f"{tick:.1f}" for tick in mve_ticks])
        ax_temp_mve.set_ylabel("Temperature MVE (remapped)", fontsize=35)
        ax_temp_mve.yaxis.set_label_position('left')
        ax_temp_mve.yaxis.tick_left()
        ax_temp_mve.set_xticks(major_ticks, minor=False)
        ax_temp_mve.xaxis.set_minor_locator(FixedLocator(x_coords))
        ax_temp_mve.tick_params(axis='y', which='major', left=True, right=False,
                    length=12, width=3.0)
        ax_temp_mve.tick_params(axis='y', which='minor', left=True, right=False,
                    length=8, width=2.0)
        ax_temp_mve.tick_params(axis='x', which='both', top=True, bottom=False,
                                labelbottom=False, labeltop=False, direction='in')
        ax_temp_mve.spines['top'].set_visible(True)
        ax_temp_mve.spines['bottom'].set_visible(False)
        ax_temp_mve.spines['right'].set_visible(False)
        ax_temp_mve.spines['left'].set_visible(False)

        output_path = os.path.join(plots_dir, "hyperparameter_contour_soft_large.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to {output_path}")


# ==========================================
# SHARED LEGEND GENERATOR
# ==========================================
def generate_shared_legend(filename, use_lines=True, markers=None):
    print(f"--- Generating Shared Legend: {filename} ---")
    legend_handles = []
    markersize_for_legend = np.sqrt(50)

    for est in legend_order:
        if est not in colors:
            continue

        if use_lines:
            ls = '--' if 'Unscaled' in est or 'Scaled' in est else '-'
            if est in ["ZGR", "ST", "STGS", "GRMC_10", "GRMC_100"]:
                handle = Line2D([0], [0], linestyle='none', marker='o',
                                markersize=markersize_for_legend, color=colors[est], label=est)
            else:
                handle = Line2D([0], [0], linestyle=ls, linewidth=2, color=colors[est], label=est)
        else:
            if markers is None:
                markers = {}
            handle = Line2D([0], [0], linestyle='none', marker=markers.get(est, 'o'),
                            markersize=markersize_for_legend, color=colors[est], label=est)

        legend_handles.append(handle)

    fig_legend = plt.figure(figsize=(9, 0.7))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles=legend_handles, ncol=len(legend_handles),
                     loc='center', frameon=False, fontsize='x-large')
    ax_legend.axis('off')

    legend_path = os.path.join(plots_dir, filename)
    fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=False)
    plt.close(fig_legend)
    print(f"Legend saved: {legend_path}")


# ==========================================
# 5. LATEX TABLE GENERATION
# ==========================================
def generate_latex_tables():
    print("\n--- Generating LaTeX Tables from output.txt ---")
    output_txt_path = os.path.join(script_dir, "output.txt")
    if not os.path.exists(output_txt_path):
        print(f"File not found: {output_txt_path}. Please run analyze_results.py > output.txt first.")
        return

    with open(output_txt_path, 'r') as f:
        content = f.read()

    sections = content.split('===')

    def display_name(est):
        name_map = {
            'ST': 'ST',
            'ZGR': 'ZGR',
            'gumbel_rao(exp_STGS)': 'Gumbel-Rao Exp',
            'reinmax(t=1.2)': r'ReinMax ($t=1.2$)',
            'reinmax(t=1.4)': r'ReinMax ($t=1.4$)',
            'MVE(exp_MVE)': 'MVE Exp',
            'MVE(t=1e6)': r'MVE ($t=10^6$)',
        }
        return name_map.get(est, est)

    def is_mve_estimator(est):
        return est.startswith('MVE')

    def format_lr(lr_val):
        """Format a learning rate as a clean scientific notation string for LaTeX."""
        if lr_val == 0:
            return "0"
        exponent = int(math.floor(math.log10(abs(lr_val))))
        mantissa = lr_val / (10 ** exponent)
        if abs(mantissa - round(mantissa)) < 1e-9:
            mantissa = int(round(mantissa))
            if mantissa == 1:
                return r"10^{" + str(exponent) + "}"
            return str(mantissa) + r"\times 10^{" + str(exponent) + "}"
        else:
            return f"{mantissa:.1f}" + r"\times 10^{" + str(exponent) + "}"

    latex_output = ""

    for i, section in enumerate(sections):
        if 'Results' not in section:
            continue
        title = section.strip()

        if i + 1 >= len(sections):
            continue

        lines = sections[i + 1].split('\n')
        table_lines = []
        in_table = False
        for line in lines:
            if line.startswith('Estimator'):
                in_table = True
            if in_table:
                if line.startswith('Missing') or line.startswith('Loaded') or (not line.strip() and len(table_lines) > 2):
                    break
                table_lines.append(line)

        if not table_lines:
            continue

        header_parts = table_lines[0].split()
        columns = header_parts[1:]

        data = {}
        data_lrs = {}
        for line in table_lines[2:]:
            if not line.strip() or line.startswith('-'):
                continue
            parts = line.split()
            est = parts[0]
            vals = []
            lrs = []
            for token in parts[1:]:
                if token == 'MISSING':
                    vals.append(float('inf'))
                    lrs.append(None)
                elif '|lr=' in token:
                    elbo_part, lr_part = token.split('|', 1)
                    vals.append(float(elbo_part))
                    lr_val = lr_part.replace('lr=', '')
                    lrs.append(float(lr_val))
                elif re.match(r'^-?\d+\.\d+$', token):
                    vals.append(float(token))
                    lrs.append(None)
                elif token.startswith('lr='):
                    lr_val = token.replace('lr=', '')
                    if lrs and lrs[-1] is None:
                        lrs[-1] = float(lr_val)
            data[est] = vals
            data_lrs[est] = lrs

        if not data:
            continue

        def col_key(c):
            dims = c.split('x')
            return (-int(dims[0]), int(dims[1]))

        sorted_indices = sorted(range(len(columns)), key=lambda idx: col_key(columns[idx]))
        sorted_columns = [columns[idx] for idx in sorted_indices]

        order = [
            'ST', 'gumbel_rao(exp_STGS)', 'ZGR',
            'reinmax(t=1.2)', 'reinmax(t=1.4)',
            'MVE(exp_MVE)', 'MVE(t=1e6)',
        ]
        sorted_ests = []
        for o in order:
            if o in data:
                sorted_ests.append(o)
        for e in data:
            if e not in sorted_ests:
                sorted_ests.append(e)

        col_mins = [float('inf')] * len(sorted_columns)
        for est in sorted_ests:
            for j, idx in enumerate(sorted_indices):
                if idx < len(data[est]):
                    val = data[est][idx]
                    if val < col_mins[j]:
                        col_mins[j] = val

        all_lr_values = set()
        for est in sorted_ests:
            if est in data_lrs:
                for lr in data_lrs[est]:
                    if lr is not None:
                        all_lr_values.add(lr)
        sorted_unique_lrs = sorted(all_lr_values)
        lr_to_symbol = {}
        for idx_lr, lr_val in enumerate(sorted_unique_lrs):
            lr_to_symbol[lr_val] = chr(ord('a') + idx_lr)

        label_safe = re.sub(r'[^a-zA-Z0-9_]', '_', title).lower()

        latex = "\\begin{table}[t]\n"
        latex += "    \\centering\n"
        title_friendly = title.replace('Results', '').strip()
        latex += "    \\caption{Final NELBO averaged over 5 seeds for " + title_friendly + "."
        if sorted_unique_lrs:
            lr_legend_parts = []
            for lr_val in sorted_unique_lrs:
                sym = lr_to_symbol[lr_val]
                lr_legend_parts.append("$^{" + sym + "}$\\,lr$={}$" + format_lr(lr_val))
            latex += " " + "; ".join(lr_legend_parts) + "."
        latex += "}\n"
        latex += "    \\label{tab:final_elbo_" + label_safe + "}\n"
        latex += "    {\n"
        latex += "    \\setlength{\\tabcolsep}{4pt}\n"
        latex += "    \\small\n"
        latex += "    \\begin{tabular}{l" + "c" * len(sorted_columns) + "}\n"
        latex += "        \\toprule\n"

        col_header_parts = []
        for c in sorted_columns:
            col_tex = c.replace('x', r'\times')
            col_header_parts.append("\\multicolumn{1}{c}{$" + col_tex + "$}")
        col_headers = " & ".join(col_header_parts)
        latex += "        Latents $V \\times K$ & " + col_headers + " \\\\\n"
        latex += "        \\midrule\n"

        printed_midrule = False

        for est in sorted_ests:
            if is_mve_estimator(est) and not printed_midrule:
                latex += "        \\midrule\n"
                printed_midrule = True

            d_name = display_name(est)

            if is_mve_estimator(est):
                row_str = "        \\textbf{" + d_name + "}"
            else:
                row_str = "        " + d_name

            for j, idx in enumerate(sorted_indices):
                if idx < len(data[est]):
                    val = data[est][idx]
                else:
                    val = float('inf')

                lr_sup = ""
                if est in data_lrs and idx < len(data_lrs[est]):
                    cell_lr = data_lrs[est][idx]
                    if cell_lr is not None and cell_lr in lr_to_symbol:
                        lr_sup = "^{" + lr_to_symbol[cell_lr] + "}"

                if val == float('inf'):
                    row_str += " & {--}"
                elif val == col_mins[j]:
                    val_str = f"{val:.1f}"
                    row_str += " & $\\mathbf{" + val_str + "}" + lr_sup + "$"
                else:
                    val_str = f"{val:.1f}"
                    row_str += " & $" + val_str + lr_sup + "$"

            row_str += " \\\\\n"
            latex += row_str

        latex += "        \\bottomrule\n"
        latex += "    \\end{tabular}\n"
        latex += "    }\n"
        latex += "\\end{table}\n\n"

        latex_output += latex

    out_path = os.path.join(plots_dir, "latex_tables.tex")
    with open(out_path, "w") as f:
        f.write(latex_output)

    print(latex_output)
    print(f"Saved LaTeX tables to {out_path}")


if __name__ == "__main__":
    # plot_greedy_results()
    # plot_bias_variance_scaled()
    # plot_magnitude_bias_variance()
    # plot_bias_variance_angles()
    plot_reinmax_hyperparameters()
    # generate_latex_tables()
    print("\nAll plotting tasks completed.")