from __future__ import annotations

import argparse
import pandas as pd
from importlib import import_module
from pathlib import Path
import sys

if __package__:
    create_manifest = import_module(f"{__package__}.create_manifest")
    convert_data = import_module(f"{__package__}.convert_data")
    top_level_files = import_module(f"{__package__}.top_level_files")
else:
    this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir))
    create_manifest = import_module("create_manifest")
    convert_data = import_module("convert_data")
    top_level_files = import_module("top_level_files")


def load_manifest() -> pd.DataFrame:
    manifest_path = convert_data.MANIFEST_CSV
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest CSV does not exist: {manifest_path}. "
            "Run create_manifest.py first or use --run-manifest-step."
        )
    return pd.read_csv(manifest_path)


def apply_runtime_config(args: argparse.Namespace) -> None:
    convert_data.DRY_RUN = args.dry_run
    convert_data.OVERWRITE = args.overwrite_data
    convert_data.SKIP_DATA_COPY = args.metadata_only_overwrite

    top_level_files.OVERWRITE = args.overwrite_top_level

    if args.metadata_only_overwrite:
        # In metadata-only mode, always refresh top-level + sidecar metadata while
        # leaving data payload files/directories untouched.
        convert_data.OVERWRITE = True
        top_level_files.OVERWRITE = True


def run_pipeline(
    run_manifest_step: bool,
    run_top_level_step: bool,
    run_data_conversion_step: bool,
) -> None:
    if run_manifest_step:
        print("[1/3] Creating source manifest...")
        create_manifest.create_manifest()

    manifest = load_manifest()

    if run_top_level_step:
        print("[2/3] Writing top-level BIDS files...")
        top_level_files.write_top_level_files(manifest)
        top_level_files.write_derivative_dataset_descriptions()

    if run_data_conversion_step:
        print("[3/3] Converting and copying manifest rows...")
        convert_data.run_data_conversion(manifest)

    print("Pipeline finished.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BIDS conversion pipeline with centralized step/config flags.",
    )

    parser.add_argument(
        "--run-manifest-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate source_manifest.csv before conversion (default: true).",
    )
    parser.add_argument(
        "--run-top-level-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write dataset_description/README/participants/sessions files (default: true).",
    )
    parser.add_argument(
        "--run-data-conversion-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy/convert manifest rows and write sidecars/logs (default: true).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write/copy files in data conversion step.",
    )
    parser.add_argument(
        "--overwrite-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow data conversion to overwrite existing outputs (default: false).",
    )
    parser.add_argument(
        "--overwrite-top-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow top-level file generation to overwrite existing outputs (default: true).",
    )

    parser.add_argument(
        "--metadata-only-overwrite",
        action="store_true",
        help=(
            "Rewrite top-level files and sidecars/json/tsv metadata while skipping all data payload copies. "
            "This implies overwrite behavior for metadata outputs."
        ),
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    apply_runtime_config(args)

    run_pipeline(
        run_manifest_step=args.run_manifest_step,
        run_top_level_step=args.run_top_level_step,
        run_data_conversion_step=args.run_data_conversion_step,
    )


if __name__ == "__main__":
    main()
