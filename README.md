# GeoMetaMorph Framework

Abstract base classes for GeoMetaMorph.

## Modules

```
framework/
├── __init__.py      # exports AbstractRunner, AbstractEvaluator, DEFAULT_TRANSFORMS
├── runner.py        # AbstractRunner: experiment loop
├── evaluator.py     # AbstractEvaluator: result analysis and plotting
└── transforms.py    # built-in 2D/3D numpy-array transforms
```

## AbstractRunner (`runner.py`)

Handles the full experiment loop as presented in GeoMetaMorph's prototypical implementations: checkpoint loading/saving,
transform dispatch, and failure logging.
Subclasses implement four abstract hooks and optionally override additional configurable hooks.

**Abstract hooks (must implement):**

| Hook                    | Return type          | Purpose                                                                                 |
|-------------------------|----------------------|-----------------------------------------------------------------------------------------|
| `checkpoint_filename()` | `str`                | File name for the checkpoint output file                                                |
| `list_input_files()`    | `list[(name, path)]` | Discover all input files; `name` becomes the sample identifier in all signatures        |
| `load_input(file_path)` | domain object        | Load a single input file; the returned object must be safe for repeated transform calls |
| `run_model(data)`       | `float`              | Run the domain model and return a scalar metric                                         |

**Configurable hooks (optional overrides):**

| Hook                                  | Default                              | Purpose                                                     |
|---------------------------------------|--------------------------------------|-------------------------------------------------------------|
| `transform_functions` (property)      | `DEFAULT_TRANSFORMS`                 | Dict of `{name: callable}` for all transforms               |
| `get_background(data)`                | `0`                                  | Fill value for pixels/voxels introduced by a transform      |
| `apply_transform(data, name, params)` | dispatches via `transform_functions` | Override for non-array data (e.g., ASE Atoms)               |
| `get_baseline_metric(file_path)`      | `None` (skipped)                     | Compute and store baseline metric before the transform loop |
| `log_failure(signature, error)`       | appends to `failed.txt`              | Custom failure handling                                     |

**Shared logic (no override needed):**
- `make_signature(name, transform, params)`, canonical key format: `{name}_{Transform}({p1,p2,...})`
- `load_checkpoint()` / `save_checkpoint()`, persist results to a Python-literal text file
- `run()`, sequential loop over all files x transforms x parameter vectors; already-computed entries are skipped

Parallelization is intentionally left to the subclass: override `run()` or wrap the inner body in a `Pool` or `ProcessPoolExecutor`.

**Minimal skeleton:**

```python
from framework import AbstractRunner

class MyRunner(AbstractRunner):
    def checkpoint_filename(self):
        return "results.txt"

    def list_input_files(self):
        return [(Path(f).stem, os.path.join(self.input_path, f))
                for f in os.listdir(self.input_path) if f.endswith(".dat")]

    def load_input(self, file_path):
        return np.load(file_path)

    def run_model(self, data):
        return my_model(data)  # returns a float

if __name__ == "__main__":
    MyRunner(INPUT_DIR, OUTPUT_DIR, PARAMS).run()
```

## AbstractEvaluator (`evaluator.py`)

Handles result parsing, relative-error computation, grouping, and boxplot generation.
Subclasses implement two abstract hooks.

**Abstract hooks (must implement):**

| Hook                      | Return type               | Purpose                                        |
|---------------------------|---------------------------|------------------------------------------------|
| `parse_output()`          | `dict[signature, metric]` | Read the checkpoint file(s) into a flat dict   |
| `extract_originals(data)` | `dict[name, baseline]`    | Identify baseline metrics from the parsed data |

**Shared logic (no override needed):**
- `compute_relative_errors`: `|transformed - original| / |original|` per signature
- `group_by_transform_and_param`: groups errors by transform type and parameter combination
- `sort_params`: sorts parameter-combination keys numerically
- `split_by_axis`: for `Translate*` and `Rotate*`, partitions results into per-axis sub-groups 
   (`_x`, `_y`, `_z`, `_mixed`); dimensionality (2D vs 3D) is inferred at runtime from the parameter count
- `create_boxplots`: one PNG per transform group, symmetric-log y-axis, configurable threshold line (default 5 %);
   large groups are split across multiple files
- `run()`: full pipeline: parse -> extract originals -> errors -> group -> sort/split -> plot

**Minimal skeleton:**

```python
from framework import AbstractEvaluator

class MyEvaluator(AbstractEvaluator):
    def parse_output(self):
        with open(os.path.join(self.output_path, "results.txt")) as f:
            return ast.literal_eval(f.read())

    def extract_originals(self, data):
        return {k: v for k, v in data.items() if "_" not in k}

if __name__ == "__main__":
    MyEvaluator("output/", "plots/", label="MyTool").run()
```

## Built-in transforms (`transforms.py`)

All functions share the signature `func(image, *params, background=0) -> ndarray` and never modify the input in-place.
`scikit-image` is used for scaling when available, with a `scipy.ndimage.zoom` fallback.

| Name          | Params                      | Notes                                  |
|---------------|-----------------------------|----------------------------------------|
| `Translate2D` | `dx, dy`                    | pixel shift                            |
| `Rotate2D`    | `angle`                     | counter-clockwise, degrees             |
| `Mirror2D`    | `axis`                      | 0 = vertical flip, 1 = horizontal flip |
| `Scale2D`     | `factor`                    | uniform scale                          |
| `Translate3D` | `dx, dy, dz`                | voxel shift                            |
| `Rotate3D`    | `angle_x, angle_y, angle_z` | sequential z→y→x rotation              |
| `Mirror3D`    | `axis`                      | 0, 1, or 2                             |
| `Scale3D`     | `factor`                    | uniform scale                          |

To extend the registry without replacing the defaults:

```python
@property
def transform_functions(self):
    return {**DEFAULT_TRANSFORMS, "MyTransform3D": my_fn}
```

### Default parameters (`DEFAULT_PARAMS`)

`DEFAULT_PARAMS` is a ready-made parameter dict for `AbstractRunner`:

**Selection rules:**
- **2D transforms** (`Mirror2D`, `Scale2D`, `Translate2D`): parameter values rated Stable in PolarityJaM.
- **3D transforms** (`Mirror3D`, `Scale3D`, `Rotate3D`): intersection of parameter values rated Stable in both PS3D and GPAW.
- `Rotate2D` is omitted as Rotate z is Unstable in PolarityJaM.
- `Translate3D` uses PS3D stable values only: GPAW translations are Unsafe due to that domain's geometry optimizer, not because translation is generally unsafe.

```python
from framework import DEFAULT_PARAMS

runner = MyRunner(INPUT_DIR, OUTPUT_DIR, DEFAULT_PARAMS)
```

## Checkpoint format

Results are stored as a plain Python dict literal (readable with `ast.literal_eval`):

```
{'Sample1': 42.0, 'Sample1_Translate3D(10,0,0)': 41.3, 'Sample1_Mirror3D(0)': 42.1, ...}
```

Baseline entries use the bare sample name as key. The file is written after every completed experiment, so runs can be safely interrupted and resumed.

## Experiment parameters format

The `parameters` argument to `AbstractRunner` maps each transform name to a list of parameter vectors:

```python
PARAMS = {
    "Translate3D": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
    "Rotate3D":    [[15, 0, 0], [0, 15, 0], [0, 0, 15]],
    "Mirror3D":    [[0], [1], [2]],
    "Scale3D":     [[0.9], [1.1], [1.5]],
}
```