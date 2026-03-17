"""
framework/evaluator.py
======================
Abstract base class for GeoMetaMorph result evaluators.

Domain scientists subclass AbstractEvaluator and implement two hooks:

    parse_output()          - read experiment output -> {signature: metric}
    extract_originals(data) - identify baseline metrics -> {name: metric}

The base class handles:
    - computation of relative-errors
    - grouping by transformation type
    - numeric sorting of parameter combinations
    - per-axis splitting for multi-axis transforms (e.g., Translate* or Rotate*)
    - boxplot generation with scale=symlog and configurable stability-threshold line

Supports 2D (e.g. Translate2D with (x, y) params) and 3D (e.g. Translate3D with (x, y, z) params) transparently; all
axis labels are inferred at runtime from the number of parameters in each signature, such that no dimensionality has to
be declared by the subclass.

Example skeleton
----------------
class MyEvaluator(AbstractEvaluator):
    def parse_output(self):
        with open(os.path.join(self.output_path, "results.txt")) as f:
            return ast.literal_eval(f.read())

    def extract_originals(self, data):
        return {k: v for k, v in data.items() if "_" not in k}

if __name__ == "__main__":
    MyEvaluator("output/", "plots/", "MyTool").run()
"""

import os
import re
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatter, FuncFormatter


class AbstractEvaluator(ABC):
    """
    Template for evaluating geometric-transform sensitivity experiments.

    Parameters
    ----------
    output_path : str
        Directory containing the experiment output (checkpoint file, etc.).
    plot_output_dir : str
        Directory where PNG boxplots will be saved.
    label : str
        Short identifier appended to plot filenames (e.g. ``"GPAW"``).
    threshold : float
        Value at which a horizontal reference line is drawn on every plot.
        Defaults to 0.05 (5 % relative error).
    """

    # Name prefixes for transforms where parameters span multiple independent axes.
    MULTI_AXIS_TRANSFORMS: tuple = ("Translate", "Rotate")

    def __init__(
        self,
        output_path: str,
        plot_output_dir: str,
        label: str,
        threshold: float = 0.05,
    ):
        self.output_path = output_path
        self.plot_output_dir = plot_output_dir
        self.label = label
        self.threshold = threshold

    # ── abstract hooks ──────────────────────────────────────────────────────────

    @abstractmethod
    def parse_output(self) -> dict:
        """
        Read the experiment result file(s) and return a flat dict mapping
        each signature string to its scalar metric, e.g.
        {
            SampleName_TransformationType(x1,y1,z1): transformed_output1,
            SampleName_TransformationType(x2,y2,z2) : transformed_output2
        }.
        """

    @abstractmethod
    def extract_originals(self, data: dict) -> dict:
        """
        Given the full parsed data dict, return a mapping of
        {sample_name: baseline_metric}
        for every sample that has a baseline (unmodified) measurement.
        """

    ####optional hook###################################################################################################

    def calculate_relative_errors(self, original: float, transformed: float) -> float:
        """Return |transformed - original| / |original|."""
        return abs(transformed - original) / abs(original)

    ####concrete shared logic###########################################################################################

    @staticmethod
    def extract_param_numbers(label: str) -> tuple:
        """
        Pull numeric values from the parameter portion of a signature label.

        Examples
        --------
        ``"CaO_Translate3D(10,0,0)"``  -> ``(10.0, 0.0, 0.0)``
        ``"Ore5_Rotate3D(0;5;0)"``     -> ``(0.0, 5.0, 0.0)``
        ``"img_Translate2D(3,-1)"``    -> ``(3.0, -1.0)``
        """
        match = re.search(r'\(([^)]+)\)', label)
        if match:
            return tuple(map(float, re.findall(r'-?\d+\.?\d*', match.group(1))))
        return ()

    @staticmethod
    def _transform_type(key: str) -> str | None:
        """
        Extract the transformation name from a signature key, or returns None if the key does not contain a recognized
        transformation name. (The transform name is the underscore-separated segment that begins with Scale, Translate,
        Rotate, or Mirror.)
        """
        for t in ("Scale", "Translate", "Rotate", "Mirror"):
            if t in key:
                segment = next(
                    (part for part in key.split("_") if part.startswith(t)),
                    None,
                )
                if segment:
                    # Strip parameter portion to get the bare transform name
                    return segment.split("(")[0]
        return None

    def compute_relative_errors(self, data: dict, originals: dict) -> dict:
        """
        For every entry in the data dict that has a corresponding baseline in the originals dict, compute the rel. error
        and return a new dict {signature: relative_error}. Note: Entries without a "_" in their key (bare sample names)
        are skipped, as are entries whose sample is not in the originals dict.
        """
        errors = {}
        for key, value in data.items():
            if "_" not in key:
                continue
            name = key.split("_")[0]
            if name not in originals:
                continue
            errors[key] = self.calculate_relative_error(originals[name], value)
        return errors

    def group_by_transform_and_param(self, errors: dict) -> dict:
        """
        Groups {signature: error} into
        {TransformName:
            {full_param_str:
                [error, ...]
            }
        }

        The param string is the portion of the signature after the sample name (e.g. "Sample_Translate3D(10,0,0)" ->
        "Translate3D(10,0,0)").
        """
        grouped: dict = {}
        for key, value in errors.items():
            transform = self._transform_type(key)
            if transform is None:
                continue
            # param_str: everything after the first underscore
            param_str = key.split("_", 1)[1]
            grouped.setdefault(transform, {}).setdefault(param_str, []).append(value)
        return grouped

    def sort_params(self, transform: str, param_keys: list) -> list:
        """Sort parameter-combination keys numerically."""
        return sorted(param_keys, key=lambda k: self.extract_param_numbers(k))

    def split_by_axis(self, transform: str, sorted_data: dict) -> dict:
        """
        For multi-axis transforms (Translate*, Rotate*), partition the sorted_data dict into per-axis sub-dicts keyed
        by "{transform}_x", "_y", "_z" (3-D) or "_x"``, "_y" (2-D), with "_mixed" for combinations where more than one
        axis is non-zero. Axis labels are inferred from the number of parameters in each signature entry, such that no
        dimensionality has to be declared.

        For transforms that are not in MULTI_AXIS_TRANSFORMS, the data is returned unchanged as {transform: sorted_data}.
        """
        if not any(transform.startswith(t) for t in self.MULTI_AXIS_TRANSFORMS):
            return {transform: sorted_data}

        axes: dict = {}
        for key, values in sorted_data.items():
            nums = self.extract_param_numbers(key)
            if not nums:
                axes.setdefault(transform, {})[key] = values
                continue

            # Build axis label list matching the parameter dimensionality
            dim = len(nums)
            axis_labels = ["x", "y", "z"][:dim]

            nonzero = [i for i, v in enumerate(nums) if v != 0.0]
            if len(nonzero) == 1:
                bucket = f"{transform}_{axis_labels[nonzero[0]]}"
            else:
                # All-zero or mixed
                bucket = f"{transform}_mixed"

            axes.setdefault(bucket, {})[key] = values

        return axes if axes else {transform: sorted_data}

    def prepare_plot_data(self, grouped: dict) -> dict:
        """
        Sort parameter keys and split multi-axis transforms into per-axis sub-dicts. Returns a flat dict ready for
        create_boxplots().
        """
        result: dict = {}
        for transform, params in grouped.items():
            sorted_keys = self.sort_params(transform, list(params.keys()))
            sorted_data = {k: params[k] for k in sorted_keys}
            result.update(self.split_by_axis(transform, sorted_data))
        return result

    @staticmethod
    def _custom_formatter(value, _):
        """Human-readable y-axis labels: 4 decimal places between 0 and 1."""
        if value == 0.0:
            return f"{value:g}"
        if 0.0 < abs(value) < 1.0:
            return f"{value:.4f}"
        return f"{value:g}"

    def create_boxplots(self, plot_data: dict, max_per_plot: int = 52) -> None:
        """
        Generate one (or more) PNG boxplot(s) per transformation group.

        Each uses symmetric-log y-axis and displays a horizontal reference line at self.threshold. If a group has more
        than max_per_plot parameter combinations, it is split across multiple files.

        Files are saved to self.plot_output_dir following the {transform}_{i}_{label}.png naming format.
        """
        os.makedirs(self.plot_output_dir, exist_ok=True)

        for transform, params in plot_data.items():
            if not params:
                continue

            items = list(params.items())
            num_plots = max(1, (len(items) + max_per_plot - 1) // max_per_plot)

            for i in range(num_plots):
                chunk = items[i * max_per_plot: (i + 1) * max_per_plot]
                data = [v for _, v in chunk]
                labels = [k for k, _ in chunk]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title(transform)
                ax.boxplot(data, labels=labels)
                ax.axhline(
                    y=self.threshold,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                )
                ax.set_yscale("symlog", linthresh=1e-3)
                ax.yaxis.set_major_formatter(
                    FuncFormatter(self._custom_formatter)
                )

                # Secondary y-axis showing threshold label
                rax = ax.secondary_yaxis("right")
                rax.set_yscale("symlog", linthresh=1e-3)
                rax.yaxis.set_major_locator(FixedLocator([self.threshold]))
                rax.set_yticks([self.threshold])
                rax.set_yticklabels([f"{self.threshold} (Threshold)"])

                ax.set_xlabel("Parameter Combinations")
                ax.set_ylabel("Relative Error (log)")
                plt.xticks(rotation=90)
                plt.tight_layout()

                suffix = f"_{i + 1}" if num_plots > 1 else ""
                filename = f"{transform}{suffix}_{self.label}.png"
                plt.savefig(os.path.join(self.plot_output_dir, filename))
                plt.close(fig)

    def run(self) -> None:
        """
        Full evaluation pipeline:
        parse -> extract originals -> compute errors -> group -> sort/split -> plot.
        """
        data = self.parse_output()
        originals = self.extract_originals(data)
        errors = self.compute_relative_errors(data, originals)
        grouped = self.group_by_transform_and_param(errors)
        plot_data = self.prepare_plot_data(grouped)
        self.create_boxplots(plot_data)