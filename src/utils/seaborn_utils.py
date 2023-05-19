import math
import pandas as pd
# pylint: disable=import-error
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def create_lineplot_for_runtime(df: pd.DataFrame, pass_col: str = "Cmb", save_plot=None,
                                show=False):
    """Create a lineplot for Runtime (ms) vs Sequence Length (number of tokens).
    Coutsey ChatGPT ^_^

    Args:
        df (pd.DataFrame): The DataFrame object containing the runtime data.
            df.index = "seq_len": Sequence Lengths that each measurement was run.
            df.columns = ["pass", "methods"]: A hierarchical column header structure.
                - "pass": fwd, bwd, cmb for Forward-pass, Backward-pass, Combined
                  measurements, respectively
                - "methods": identifier to all the methods (include baselines) measured.
            The runtime data is in seconds.
        pass_col (str): Select which pass to plot.
    
    By default, the plot is in log-log scale.
    """

    # Filter the DataFrame to include only columns with header 'cmb'
    df_cmb = df.loc[:, (pass_col, slice(None))]
    df_cmb *= 1000

    # Reshape the DataFrame for lineplot
    df_melted = df_cmb.stack(level='methods').reset_index()

    # Set up the plot
    sns.set(style='whitegrid')
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot the lines with hue
    sns.lineplot(data=df_melted, x='seq_len', y=pass_col, hue='methods', ax=ax,
                 markers=True, dashes=False, palette='Set1')

    # Set the x-axis to log2 scale
    ax.set_xscale('log', base=2)

    # Set the y-axis to log10 scale
    ax.set_yscale('log', base=10)

    # Set plot title and axis labels
    ax.set_title('Fwd + Bwd Pass', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Sequence Length', fontsize=20)
    ax.set_ylabel('Runtime (ms)', fontsize=20)

    # Customize tick labels and font sizes
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.index, fontsize=20)
    ax.tick_params(axis='y', which="major", labelsize=20)

    # Set the tick direction to outward for the left and bottom axes
    ax.tick_params(axis='both', left=True, bottom=True, direction='out',
                   length=8, width=2)
    ax.tick_params(axis="y", which="minor", left=True, direction="out")

    # Add a margin between the lowest y-tick and the bottom spine
    bottom_margin = ax.get_ylim()[0] * 0.3  # Adjust the margin
    ax.set_ylim(bottom=ax.get_ylim()[0] - bottom_margin)

    # Create a separate legend outside of the axes
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                        title='Method', borderaxespad=0.5)

    # Set legend font size
    legend.get_title().set_fontsize(12)

    # Set the plot frame to be bold and thicker
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Set the plot frame color to black
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # Adjust layout
    plt.subplots_adjust(bottom=0.5, left=0.3)

    # Save the plot as PNG image
    if save_plot is not None:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')

    # Adjust layout and display the plot
    if show:
        plt.tight_layout()
        plt.show()


def create_lineplot_for_memory(df: pd.DataFrame, pass_col: str = "Cmb", save_plot=None,
                                show=False):
    """Create a lineplot for Memory (GB) vs Sequence Length (number of tokens).
    Coutsey ChatGPT ^_^

    Args:
        df (pd.DataFrame): The DataFrame object containing the runtime data.
            df.index = "seq_len": Sequence Lengths that each measurement was run.
            df.columns = ["pass", "methods"]: A hierarchical column header structure.
                - "pass": fwd, bwd, cmb for Forward-pass, Backward-pass, Combined
                  measurements, respectively
                - "methods": identifier to all the methods (include baselines) measured.
            The runtime data is in seconds.
        pass_col (str): Select which pass to plot.
    
    By default, the plot is in log-linear scale.
    """

    # Filter the DataFrame to include only columns with header 'cmb'
    df_cmb = df.loc[:, (pass_col, slice(None))]

    # Reshape the DataFrame for lineplot
    df_melted = df_cmb.stack(level='methods').reset_index()

    # Set up the plot
    sns.set(style='whitegrid')
    _, ax = plt.subplots(figsize=(8, 6))

    # Plot the lines with hue
    sns.lineplot(data=df_melted, x='seq_len', y=pass_col, hue='methods', ax=ax,
                 markers=True, dashes=False, palette='Set1')

    # Set the x-axis to log2 scale
    #ax.set_xscale('log', base=2)

    # Set plot title and axis labels
    ax.set_title('Fwd + Bwd Pass', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Sequence Length', fontsize=20)
    ax.set_ylabel('Memory Footprint (GB)', fontsize=20)

    # Function to format tick labels
    def log2_tick_formatter(x, pos):
        if x <= 512:
            return x
        else:
            power = int(math.log2(x))
            return f"{2 ** (power-10):.0f}k"

    # Set custom tick formatter for x-axis

    # Customize tick labels and font sizes
    ax.set_xticks(df.index)
    ax.xaxis.set_major_formatter(FuncFormatter(log2_tick_formatter))
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Set the tick direction to outward for the left and bottom axes
    ax.tick_params(axis='both', left=True, bottom=True, direction='out',
                   length=8, width=2)

    # Add a margin between the lowest y-tick and the bottom spine
    bottom_margin = ax.get_ylim()[0] * 0.3  # Adjust the margin
    ax.set_ylim(bottom=ax.get_ylim()[0] - bottom_margin)

    # Create a separate legend outside of the axes
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                        title='Method', borderaxespad=0.5)

    # Set legend font size
    legend.get_title().set_fontsize(12)

    # Set the plot frame to be bold and thicker
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Set the plot frame color to black
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # Adjust layout
    plt.subplots_adjust(bottom=0.5, left=0.3)

    # Save the plot as PNG image
    if save_plot is not None:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')

    # Adjust layout and display the plot
    if show:
        plt.tight_layout()
        plt.show()