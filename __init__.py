"""
GeoMetaMorph framework
======================
Abstract base classes for geometric-transform sensitivity experiments.

Usage
-----
from framework import AbstractRunner, AbstractEvaluator

Full experimental run — call order
-----------------------------------
The experiment is split into two phases: running (AbstractRunner) and evaluation (AbstractEvaluator).

Phase 1 – Running (AbstractRunner.run())
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For every input file and every (transform, params) combination:

    Step 0 – Data input and preprocessing
        data = runner.load_input(file_path)
        # Converts the domain-specific file format (e.g. .nii.gz, .tif, .gpw) into
        # the internal data structure used throughout the run (typically a numpy.ndarray).
        # Only requires a custom implementation when the domain does not natively use
        # numpy.ndarrays (e.g. ASE Atoms for crystallography).

    Step 1 – Generate transformed inputs
        transformed = runner.apply_transform(data, transform_name, params)
        # Applies one geometric transformation with one parameter vector to the loaded data.
        # Dispatches to DEFAULT_TRANSFORMS (2D/3D numpy-array ops) by default;
        # override apply_transform() for non-array data structures.
        # Repeated for every (transform_name, params) entry in runner.parameters.

    Step 2 – Baseline generation
        baseline = runner.get_baseline_metric(file_path)
        # Runs the Program Under Test (PUT) on the original, unmodified input and stores
        # the result as the reference metric for this sample.
        # Called once per input file, before the transform loop.
        # Optional: the default returns None (no baseline stored).

    Step 3 – Output dataset generation
        metric = runner.run_model(transformed)
        # Runs the PUT on each transformed input and records the scalar output.
        # Results are persisted to the checkpoint file after every successful call,
        # so the loop can be safely interrupted and resumed.

Phase 2 – Evaluation (AbstractEvaluator.run())
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Step 4 – Stability evaluation
        data      = evaluator.parse_output()          # read checkpoint file -> {signature: metric}
        originals = evaluator.extract_originals(data) # {sample_name: baseline_metric}
        errors    = evaluator.compute_relative_errors(data, originals)
        # Computes |transformed - original| / |original| for every (sample, transform, params) triple.

    Step 5 – Visualisation
        grouped   = evaluator.group_by_transform_and_param(errors)
        plot_data = evaluator.prepare_plot_data(grouped)  # sort params; split Translate/Rotate by axis
        evaluator.create_boxplots(plot_data)
        # Writes one PNG per transform group (symlog y-axis, threshold reference line).
"""

from .runner import AbstractRunner
from .evaluator import AbstractEvaluator
from .transforms import DEFAULT_TRANSFORMS, DEFAULT_PARAMS

__all__ = ["AbstractRunner", "AbstractEvaluator", "DEFAULT_TRANSFORMS", "DEFAULT_PARAMS"]