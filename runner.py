"""
framework/runner.py
===================
Abstract base class for GeoMetaMorph runners.

Domain scientists need to subclass the AbstractRunner and implement the following hooks:

    checkpoint_filename()   - name of the checkpoint output file
    list_input_files()      - discover all inputs to process
    load_input(file_path)   - load domain-specific data from a file
    run_model(data)         - run the domain model and return a scalar metric

The hook apply_transform() has a ready-made default implementation that dispatches to built-in 2D/3D numpy-array
transforms provided by framework.transforms (Translate, Rotate, Mirror, Scale, in 2D and 3D).
Subclasses that work with non-array data structures (e.g. ASE Atoms for crystallography) need to override
apply_transform() to use their own transformation logic.

Optional hooks allow customizing the background fill value, extending the transformation registry, logging failures,
and storing per-file baselines. Further extension points are possible but left to the users.

Parallelization is intentionally left to the subclass: override run() or wrap the inner body of run() in a
Pool/ProcessPoolExecutor as needed.

Built-in transformations
========================
The default transform registry (framework.transforms.DEFAULT_TRANSFORMS) covers:

    In 2D: Translate2D, Rotate2D, Mirror2D, (Mirror2DX, Mirror2DY,) Scale2D
    In 3D: Translate3D, Rotate3D, Mirror3D, Scale3D

To add domain-specific transformations or replace built-ins, override the transform_functions property:

    @property
    def transform_functions(self):
        return {**DEFAULT_TRANSFORMS, "MyTransform2D": my_fn}

Example skeleton
================
class MyRunner(AbstractRunner):
    def checkpoint_filename(self):
        return "results.txt"

    def list_input_files(self):
        return [(Path(f).stem, os.path.join(self.input_path, f))
                for f in os.listdir(self.input_path) if f.endswith(".dat")]

    def load_input(self, file_path):
        return np.load(file_path)         # returns a numpy array

    def run_model(self, data):
        return my_model(data)             # returns a float

    # apply_transform() uses the built-in numpy transforms automatically.
    # Override only if the data type requires custom transform logic.

if __name__ == "__main__":
    MyRunner(INPUT, OUTPUT, PARAMS).run()
"""

import ast
import os
from abc import ABC, abstractmethod

from .transforms import DEFAULT_TRANSFORMS


class AbstractRunner(ABC):
    """
    Template for running geometric-transform sensitivity experiments.

    Parameters
    ----------
    input_path : str
        Directory containing the raw input files.
    output_path : str
        Directory where the checkpoint file (and failure log) will be written.
    parameters : dict[str, list[list]]
        Mapping of transform name -> list of parameter vectors, e.g.:

            {
                "Translate3D": [[10, 0, 0], [0, 5, 0]],
                "Mirror3D":    [[0], [1], [2]],
                "Scale3D":     [[0.5], [2.0]],
            }

    num_workers : int
        Hint for subclasses that add parallelization. The base run() loop is sequential and ignores this value.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        parameters: dict,
        num_workers: int = 1,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.parameters = parameters
        self.num_workers = num_workers
        self.checkpoint_file = os.path.join(output_path, self.checkpoint_filename())
        self.checkpoints: dict = {}

    ####abstract hooks##################################################################################################

    @abstractmethod
    def checkpoint_filename(self) -> str:
        """Return the file name (not full path) for the checkpoint file."""

    @abstractmethod
    def list_input_files(self) -> list:
        """
        Return a list of (name, file_path) tuples for every input; name is used as the sample identifier in all
        signatures and checkpoints.
        """

    @abstractmethod
    def load_input(self, file_path: str):
        """
        Load and return the domain-specific data object from file_path.

        The returned object must be safe to use as input to multiple successive apply_transform() calls (i.e.,
        transform functions must not modify it in-place). For mutable structures (e.g., ASE Atoms), return a deep copy
        or implement defensive copying inside apply_transform().
        """

    @abstractmethod
    def run_model(self, data) -> float:
        """
        Run the domain-specific model/tool on data and return a scalar metric (float).
        """

    ####configurable hooks##############################################################################################

    @property
    def transform_functions(self) -> dict:
        """
        Dict mapping transform names to callables with the signature:
            func(data, *params, background=<fill_value>) -> transformed_data

        Defaults to framework.transforms.DEFAULT_TRANSFORMS, that covers standard 2D and 3D numpy-array transforms.
        Override this property to add domain-specific transforms or replace the built-ins:
            @property
            def transform_functions(self):
                return {**DEFAULT_TRANSFORMS, "MyTransform": my_fn}
        """
        return DEFAULT_TRANSFORMS

    def get_background(self, data) -> float:
        """
        Return the fill value to use for pixels/voxels introduced by translation, rotation, or scaling.

        The default is 0. Override to derive the background from the data (e.g., the average corner intensity of an
        image).
        """
        return 0

    def apply_transform(self, data, transform_name: str, params: list):
        """
        Apply transform_name with params to data and return the result.

        Default implementation dispatches to self.transform_functions, which covers all 2D and 3D numpy-array
        transformations out of the box. Override for non-array data structures (e.g., the special ASE Atoms objects):
            def apply_transform(self, atoms, transform_name, params):
                fn = {"Translate3D": self._translate_atoms, ...}[transform_name]
                return fn(atoms.copy(), *params)

        Raises
        ------
        ValueError
            If transform_name is not found in self.transform_functions.
        """
        funcs = self.transform_functions
        if transform_name not in funcs:
            raise ValueError(
                f"Unknown transform '{transform_name}'. "
                f"Available: {sorted(funcs)}. "
                f"Override apply_transform() to handle custom data types or add the transform to the\
                  transform_functions registry."
            )
        background = self.get_background(data)
        return funcs[transform_name](data, *params, background=background)

    def get_baseline_metric(self, file_path: str):
        """
        Return the metric for the unmodified input. Called once per file before the transform loop.
        """
        return self.run_model(file_path)

    def get_transformed_metrics(self, file_paths: list[str]):
        """
        Return the metrics for the modified inputs. Called once per transformation for each file.
        """
        metrics = {}
        for file_path in file_paths:
            metrics[file_path] = self.run_model(file_path)
        return metrics

    def log_failure(self, signature: str, error: Exception) -> None:
        """Append a failure record to failed.txt in the output directory."""
        failed_path = os.path.join(self.output_path, "failed.txt")
        with open(failed_path, "a") as f:
            f.write(f"{signature} failed: {error}\n")

    ####concrete shared logic###########################################################################################

    @staticmethod
    def make_signature(name: str, transform: str, params: list) -> str:
        """
        Build the canonical experiment signature used as a checkpoint key.

        Format: {name}_{TransformName}({p1,p2,...})

        Example: CaO_Translate3D(10,0,0)
        """
        param_str = (
            str(params)
            .replace("[", "(")
            .replace("]", ")")
            .replace(" ", "")
        )
        return f"{name}_{transform}{param_str}"

    def load_checkpoint(self) -> None:
        """Read an existing checkpoint file into self.checkpoints."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                self.checkpoints = ast.literal_eval(f.read())

    def save_checkpoint(self) -> None:
        """Persist self.checkpoints to disk."""
        os.makedirs(self.output_path, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            f.write(str(self.checkpoints))

    def run(self) -> None:
        """
        Main experiment loop (sequential).

        For each input file the data is loaded once, an optional baseline metric is stored, then every
        (transform, params) combination is evaluated. Already-computed entries are skipped via the checkpoint file, so
        the loop is safe to interrupt and resume.
        """
        os.makedirs(self.output_path, exist_ok=True)
        self.load_checkpoint()

        for name, file_path in self.list_input_files():
            ####Baseline (optional)###########################
            if name not in self.checkpoints:
                baseline = self.get_baseline_metric(file_path)
                if baseline is not None:
                    self.checkpoints[name] = baseline
                    self.save_checkpoint()

            # Load input (once per file)
            data = self.load_input(file_path)

            ####Transform loop#####################################################
            for transform, param_list in self.parameters.items():
                for params in param_list:
                    sig = self.make_signature(name, transform, params)
                    if sig in self.checkpoints:
                        continue
                    try:
                        transformed = self.apply_transform(data, transform, params)
                        metric = self.run_model(transformed)
                    except Exception as e:
                        self.log_failure(sig, e)
                        continue

                    if metric is not None:
                        self.checkpoints[sig] = metric
                        self.save_checkpoint()