"""Create a cleaned IHDP file for the DML overlap project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLE_REFERENCE_COLUMNS = ["treat", "iqsb.36"]

# This is the dummy-coded representation used in the legacy example analysis.
# It keeps one clean encoding for each baseline covariate and drops the
# duplicate/factor-coded columns that were only convenient for old R scripts.
EXAMPLE_COVARIATE_COLUMNS = [
    "bw",
    "momage",
    "nnhealth",
    "birth.o",
    "parity",
    "moreprem",
    "cigs",
    "alcohol",
    "ppvt.imp",
    "bwg",
    "female",
    "mlt.birt",
    "b.marry",
    "livwho",
    "language",
    "whenpren",
    "drugs",
    "othstudy",
    "mom.lths",
    "mom.hs",
    "mom.coll",
    "mom.scoll",
    "site1",
    "site2",
    "site3",
    "site4",
    "site5",
    "site6",
    "site7",
    "site8",
    "momblack",
    "momhisp",
    "momwhite",
    "workdur.imp",
]

EXAMPLE_DROP_COLUMNS = [
    "dose400",
    "bwg.1",
    "female.1",
    "mlt.birtF",
    "b.marryF",
    "livwhoF",
    "languageF",
    "whenprenF",
    "drugs.1",
    "othstudy.1",
    "momed4F",
    "siteF",
    "momraceF",
    "workdur.imp.1",
]

SIM_REFERENCE_COLUMNS = ["treat"]

SIM_COVARIATE_COLUMNS = [
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
    "sex",
    "twin",
    "b.marr",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
    "momwhite",
    "momblack",
    "momhisp",
]

# Backward-compatible alias used elsewhere in the repo for the example-based
# schema.
COVARIATE_COLUMNS = EXAMPLE_COVARIATE_COLUMNS


def schema_config(schema: str) -> tuple[list[str], list[str], list[str]]:
    if schema == "example":
        return EXAMPLE_REFERENCE_COLUMNS, EXAMPLE_COVARIATE_COLUMNS, EXAMPLE_DROP_COLUMNS
    if schema == "sim":
        return SIM_REFERENCE_COLUMNS, SIM_COVARIATE_COLUMNS, []
    raise ValueError(f"Unknown schema: {schema}")


def build_clean_ihdp(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    reference_columns, covariate_columns, drop_columns = schema_config(schema)
    expected = set(reference_columns + covariate_columns + drop_columns)
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"Input file is missing expected columns: {missing}")

    cleaned = df[reference_columns + covariate_columns].copy()

    # Keep reference columns as metadata. The covariate matrix itself should
    # not contain missing values.
    null_covariates = cleaned[covariate_columns].isna().sum()
    if null_covariates.any():
        bad = null_covariates[null_covariates > 0].to_dict()
        raise ValueError(f"Cleaned covariates contain missing values: {bad}")

    return cleaned


def default_input_path(schema: str) -> str:
    if schema == "example":
        return "hill_data/ihdp_example.csv"
    if schema == "sim":
        return "hill_data/ihdp_sim.csv"
    raise ValueError(f"Unknown schema: {schema}")


def default_output_path(schema: str) -> str:
    if schema == "example":
        return "processed/ihdp_example_processed.csv"
    if schema == "sim":
        return "processed/ihdp_sim_processed.csv"
    raise ValueError(f"Unknown schema: {schema}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the raw IHDP CSV. Defaults depend on --schema.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the cleaned IHDP CSV. Defaults depend on --schema.",
    )
    parser.add_argument(
        "--schema",
        choices=["example", "sim"],
        default="sim",
        help="Which IHDP source schema to clean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input or default_input_path(args.schema))
    output_path = Path(args.output or default_output_path(args.schema))

    df = pd.read_csv(input_path)
    cleaned = build_clean_ihdp(df, schema=args.schema)
    reference_columns, covariate_columns, _ = schema_config(args.schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Wrote cleaned IHDP data to {output_path}")
    print(f"Schema: {args.schema}")
    print(f"Shape: {cleaned.shape}")
    print(f"Reference columns: {reference_columns}")
    print(f"Covariate columns: {len(covariate_columns)}")


if __name__ == "__main__":
    main()
