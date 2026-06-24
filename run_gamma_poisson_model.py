"""Fit phippery's Gamma-Poisson model and export the resulting scores.

By default, this script reads ``test/pickle_data/data.phip`` and writes:

* ``test/pickle_data/data.gamma_poisson.phip`` (the complete xarray Dataset)
* ``test/pickle_data/data.gamma_poisson_mlxp.csv`` (the score matrix only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

import phippery
from phippery.modeling import gamma_poisson_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "test" / "pickle_data" / "data.phip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Gamma-Poisson background model to a phippery Dataset and "
            "export both the modeled Dataset and its -log10(p) score matrix."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input .phip pickle file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Output .phip file; defaults to <input>.gamma_poisson.phip",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output score CSV; defaults to <input>.gamma_poisson_mlxp.csv",
    )
    parser.add_argument(
        "--data-table",
        default="size_factors",
        help="Dataset layer used for fitting (default: size_factors)",
    )
    parser.add_argument(
        "--new-table-name",
        default="gamma_poisson_mlxp",
        help="Name of the fitted score layer (default: gamma_poisson_mlxp)",
    )
    parser.add_argument("--starting-alpha", type=float, default=0.8)
    parser.add_argument("--starting-beta", type=float, default=0.1)
    parser.add_argument("--trim-percentile", type=float, default=99.9)
    return parser.parse_args()


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        input_path.with_name(f"{stem}.gamma_poisson.phip"),
        input_path.with_name(f"{stem}.gamma_poisson_mlxp.csv"),
    )


def validate_input(ds: xr.Dataset, data_table: str) -> None:
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xarray.Dataset, got {type(ds).__name__}")
    if data_table not in ds:
        available = ", ".join(ds.data_vars)
        raise KeyError(f"Missing data table '{data_table}'. Available: {available}")

    layer = ds[data_table]
    if layer.dims != ("peptide_id", "sample_id"):
        raise ValueError(
            f"'{data_table}' must have dimensions ('peptide_id', 'sample_id'); "
            f"got {layer.dims}"
        )

    values = np.asarray(layer.values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"'{data_table}' must contain numeric values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"'{data_table}' contains NaN or infinite values")
    if np.any(values < 0):
        raise ValueError(f"'{data_table}' contains negative values")


def write_outputs(
    ds: xr.Dataset,
    output_dataset: Path,
    output_csv: Path,
    score_table: str,
) -> None:
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset_tmp = output_dataset.with_name(f"{output_dataset.name}.tmp")
    csv_tmp = output_csv.with_name(f"{output_csv.name}.tmp")
    try:
        phippery.dump(ds, dataset_tmp)
        ds[score_table].to_pandas().to_csv(csv_tmp, float_format="%.10g")
        dataset_tmp.replace(output_dataset)
        csv_tmp.replace(output_csv)
    finally:
        dataset_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not 0 < args.trim_percentile <= 100:
        raise ValueError("--trim-percentile must be in the interval (0, 100]")

    default_dataset, default_csv = default_output_paths(input_path)
    output_dataset = (args.output_dataset or default_dataset).resolve()
    output_csv = (args.output_csv or default_csv).resolve()
    if output_dataset == input_path or output_csv == input_path:
        raise ValueError("Output paths must not overwrite the input file")
    if output_dataset == output_csv:
        raise ValueError("--output-dataset and --output-csv must be different")

    print(f"Loading Dataset: {input_path}")
    ds = phippery.load(input_path)
    validate_input(ds, args.data_table)

    alpha, beta = gamma_poisson_model(
        ds,
        starting_alpha=args.starting_alpha,
        starting_beta=args.starting_beta,
        trim_percentile=args.trim_percentile,
        data_table=args.data_table,
        inplace=True,
        new_table_name=args.new_table_name,
    )

    ds[args.new_table_name].attrs.update(
        {
            "model": "gamma_poisson",
            "score": "-log10(p)",
            "fitted_alpha": float(alpha),
            "fitted_beta": float(beta),
            "source_data_table": args.data_table,
            "trim_percentile": float(args.trim_percentile),
            "starting_alpha": float(args.starting_alpha),
            "starting_beta": float(args.starting_beta),
        }
    )
    write_outputs(ds, output_dataset, output_csv, args.new_table_name)

    print(f"Fitted alpha: {alpha:.10g}")
    print(f"Fitted beta:  {beta:.10g}")
    print(f"Score shape:  {ds[args.new_table_name].shape}")
    print(f"Dataset:     {output_dataset}")
    print(f"CSV:         {output_csv}")


if __name__ == "__main__":
    main()
