#!/usr/bin/env python3
"""
Generate example plots for visualization documentation.
Run this script to create figures that will be embedded in the docs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the pyspc module to path (assuming this script is run from docs directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pyspc
except ImportError:
    print(
        "Could not import pyspc. Make sure the package is installed or run from "
        "the correct directory."
    )
    sys.exit(1)

# Set matplotlib style for consistent plots
plt.style.use("default")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


def generate_sample_data():
    """Generate realistic spectral data for examples."""
    # Wavelength range (e.g., Raman spectroscopy)
    wl = np.linspace(400, 1800, 300)
    n_spectra = 12

    # Create realistic spectral profiles with peaks
    base_spectrum = np.exp(-(((wl - 1000) / 200) ** 2)) + 0.5 * np.exp(
        -(((wl - 800) / 150) ** 2)
    )
    noise_level = 0.1

    spectra = []
    for i in range(n_spectra):
        # Add variation and noise
        variation = 1 + 0.3 * np.sin(i * 0.5) + 0.2 * np.random.randn()
        noise = noise_level * np.random.randn(len(wl))
        spectrum = variation * base_spectrum + noise
        spectra.append(spectrum)

    spectra = np.array(spectra)

    # Create metadata
    metadata = pd.DataFrame(
        {
            "sample": [f"S{i:02d}" for i in range(n_spectra)],
            "group": ["A", "B", "C"] * 4,
            "condition": ["control", "treated"] * 6,
            "intensity": np.random.uniform(0.5, 2.0, n_spectra),
            "concentration": [1, 2, 3, 4] * 3,
        }
    )

    return pyspc.SpectraFrame(spectra, wl=wl, data=metadata)


def create_basic_plot():
    """Create basic plotting example."""
    # Simple data for basic plot
    wl = np.linspace(400, 800, 200)
    spectra = []
    for i in range(5):
        base = np.exp(-(((wl - 600) / 100) ** 2))
        noise = 0.05 * np.random.randn(len(wl))
        spectra.append(base + noise)

    sf = pyspc.SpectraFrame(np.array(spectra), wl=wl)

    fig, axs = sf.plot()
    axs[0, 0].set_xlabel("Wavelength (nm)")
    axs[0, 0].set_ylabel("Intensity")
    axs[0, 0].set_title("Basic Spectral Plot")
    plt.tight_layout()

    plt.savefig("docs/img/basic_plot.png", bbox_inches="tight")
    plt.close()
    print("Generated: basic_plot.png")


def create_faceted_plot():
    """Create faceted plotting example."""
    sf = generate_sample_data()

    fig, axs = sf.plot(rows="condition", columns="group")
    fig.suptitle("Faceted Plot: Condition × Group", y=0.98, fontsize=14)

    # Add axis labels
    for ax in axs.flat:
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plt.savefig("docs/img/faceted_plot.png", bbox_inches="tight")
    plt.close()
    print("Generated: faceted_plot.png")


def create_color_grouping_plot():
    """Create color grouping example."""
    sf = generate_sample_data()

    fig, axs = sf.plot(colors="group", palette="Set1")
    axs[0, 0].set_xlabel("Wavenumber (cm⁻¹)")
    axs[0, 0].set_ylabel("Intensity")
    axs[0, 0].set_title("Color Grouping by Sample Group")
    plt.tight_layout()

    plt.savefig("docs/img/color_grouping.png", bbox_inches="tight")
    plt.close()
    print("Generated: color_grouping.png")


def create_palette_comparison():
    """Create palette comparison example."""
    sf = generate_sample_data()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Default palette
    fig1, axs1 = sf.plot(colors="group")
    # Extract the plot from axs1 and put it in axes[0]
    for line in axs1[0, 0].get_lines():
        axes[0].plot(
            line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha=0.8
        )
    axes[0].set_title("Default Palette")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")
    axes[0].set_ylabel("Intensity")
    plt.close(fig1)

    # Viridis palette
    fig2, axs2 = sf.plot(colors="concentration", palette="viridis")
    for line in axs2[0, 0].get_lines():
        axes[1].plot(
            line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha=0.8
        )
    axes[1].set_title("Viridis Palette")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    plt.close(fig2)

    # Custom colors
    custom_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    fig3, axs3 = sf.plot(colors="concentration", palette=custom_colors)
    for line in axs3[0, 0].get_lines():
        axes[2].plot(
            line.get_xdata(), line.get_ydata(), color=line.get_color(), alpha=0.8
        )
    axes[2].set_title("Custom Colors")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    plt.close(fig3)

    plt.tight_layout()
    plt.savefig("docs/img/palette_comparison.png", bbox_inches="tight")
    plt.close()
    print("Generated: palette_comparison.png")


def create_customization_plot1():
    """Create a plot with customizations."""
    sf = generate_sample_data()
    fig, axs = sf.plot(colors="sample", alpha=0.7, linewidth=2, linestyle="--")
    fig.suptitle("Customize line properties", fontsize=14, y=0.98)

    plt.savefig("docs/img/customization_plot1.png", bbox_inches="tight")
    plt.close()
    print("Generated: customization_plot1.png")


def create_customization_plot2():
    """Create a plot with customizations."""
    sf = generate_sample_data()
    fig, axs = sf.plot(columns="group", sharex=False, sharey=True)
    fig.suptitle("Control subplot sharing", fontsize=14, y=0.98)

    plt.savefig("docs/img/customization_plot2.png", bbox_inches="tight")
    plt.close()
    print("Generated: customization_plot2.png")


def create_customization_plot3():
    """Create a plot with customizations."""
    sf = generate_sample_data()
    fig, axs = sf.plot(colors="sample", legend=False)
    fig.suptitle("Hide legend", fontsize=14, y=0.98)

    plt.savefig("docs/img/customization_plot3.png", bbox_inches="tight")
    plt.close()
    print("Generated: customization_plot3.png")


def create_existing_figure_plot():
    """Create a plot using an existing figure."""
    sf = generate_sample_data()

    fig, _ = plt.subplots(1, 3, figsize=(10, 8), layout="tight")
    fig, axs = sf.plot(columns="group", fig=fig)
    fig.suptitle("Plot into an existing matplotlib figure")

    plt.savefig("docs/img/existing_figure_plot.png", bbox_inches="tight")
    plt.close()
    print("Generated: existing_figure_plot.png")


def create_combined_sf_plot():
    """Create a combined plot with multiple SpectraFrames."""
    sf1 = generate_sample_data()
    sf2 = sf1.copy()

    # Modify second SpectraFrame to have different metadata
    sf2.wl = sf2.wl - 300

    fig, axs = sf1.plot(columns="group", colors="condition")
    fig, axs = sf2.plot(columns="group", colors="condition", fig=fig)
    fig.suptitle("Combined SpectraFrames Plot")

    plt.savefig("docs/img/combined_sf_plot.png", bbox_inches="tight")
    plt.close()
    print("Generated: combined_sf_plot.png")


def create_combined_plot():
    """Create a comprehensive example combining all features."""
    sf = generate_sample_data()

    fig, axs = sf.plot(
        columns="group",
        colors="condition",
        palette=["#2E86AB", "#A23B72"],
        alpha=0.8,
        linewidth=1.5,
    )

    fig.suptitle("Combined Features: Facets + Colors", fontsize=14, y=0.98)

    # Customize each subplot
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        if i == 0:
            ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig("docs/img/combined_features.png", bbox_inches="tight")
    plt.close()
    print("Generated: combined_features.png")


if __name__ == "__main__":
    print("Generating visualization examples...")

    # Create output directory if it doesn't exist
    os.makedirs("docs/img", exist_ok=True)

    # Generate all example plots
    create_basic_plot()
    create_faceted_plot()
    create_color_grouping_plot()
    create_palette_comparison()
    create_customization_plot1()
    create_customization_plot2()
    create_customization_plot3()
    create_existing_figure_plot()
    create_combined_sf_plot()
    create_combined_plot()

    print("\nAll visualization examples generated successfully!")
    print("Images saved to docs/img/")
