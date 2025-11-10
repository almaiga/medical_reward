"""
Data loading and preparation from MEDEC dataset.

This module handles loading medical error data from the MEDEC dataset
and preparing it for self-play training with clean→error transformations.
"""

import pandas as pd
from datasets import Dataset, concatenate_datasets


def load_and_prepare_data(num_samples: int):
    """Loads MEDEC data with clean→error transformation approach.

    Uses only Error Flag = 1 rows (notes with errors):
    - Harmful game: Show clean note + error example → ask to introduce similar error
    - Safe game: Show clean note → ask to keep it safe

    This teaches the attacker real medical error patterns from MEDEC.
    
    Args:
        num_samples: Total number of samples to prepare (split 50/50 harmful/safe)
        
    Returns:
        (ds_seeds, ds_few_shot) tuple:
            - ds_seeds: Dataset with seed notes for training
            - ds_few_shot: Dataset with few-shot examples
    """
    print(
        "Loading and preparing MEDEC data with clean→error transformation approach..."
    )
    path = "data_copy/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv"
    df = pd.read_csv(path).fillna("")

    # Use only Error Flag = 1 (rows with errors)
    df_errors = df[df["Error Flag"] == 1].copy()

    # Filter out rows with empty text
    df_errors = df_errors[
        (df_errors["Text"].str.strip() != "")
        & (df_errors["Corrected Text"].str.strip() != "")
    ]

    print(f"📊 Available rows with errors: {len(df_errors)}")

    # Split samples 50/50 between harmful and safe
    half_samples = num_samples // 2

    # Harmful game: Show clean→error transformation
    df_harmful = df_errors.copy()
    df_harmful["seed_note"] = df_harmful[
        "Corrected Text"
    ].str.strip()  # Clean (what attacker receives)
    df_harmful["error_example"] = df_harmful[
        "Text"
    ].str.strip()  # Error version (shown as example)
    df_harmful["error_type"] = df_harmful["Error Type"]  # Type of error
    df_harmful["game_type"] = "harmful"

    # Safe game: Just keep clean
    df_safe = df_errors.copy()
    df_safe["seed_note"] = df_safe["Corrected Text"].str.strip()  # Clean
    df_safe["error_example"] = ""  # No error example for safe game
    df_safe["error_type"] = "none"
    df_safe["game_type"] = "safe"

    # Create datasets
    ds_harmful = (
        Dataset.from_pandas(
            df_harmful[["seed_note", "error_example", "error_type", "game_type"]]
        )
        .shuffle(seed=42)
        .select(range(min(half_samples, len(df_harmful))))
    )

    ds_safe = (
        Dataset.from_pandas(
            df_safe[["seed_note", "error_example", "error_type", "game_type"]]
        )
        .shuffle(seed=43)
        .select(range(min(half_samples, len(df_safe))))
    )

    # Combine both types
    ds_seeds = concatenate_datasets([ds_harmful, ds_safe]).shuffle(seed=44)

    # Few-shot examples: Show clean → error transformations with error types
    df_few_shot = df_errors.head(5).copy()
    df_few_shot["seed_note"] = df_few_shot["Corrected Text"].str.strip()  # Clean
    df_few_shot["error_example"] = df_few_shot["Text"].str.strip()  # With error
    df_few_shot["error_type"] = df_few_shot["Error Type"]

    ds_few_shot = Dataset.from_pandas(
        df_few_shot[["seed_note", "error_example", "error_type"]]
    )

    print(f"✅ Created {len(ds_harmful)} harmful + {len(ds_safe)} safe seed prompts")
    print(f"✅ Few-shot examples: {len(ds_few_shot)}")
    return ds_seeds, ds_few_shot
