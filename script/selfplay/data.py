"""
Data loading and preparation from MEDEC dataset.

This module handles loading medical error data from the MEDEC dataset
and preparing it for self-play training with clean→error transformations.
"""

import pandas as pd
from datasets import Dataset, concatenate_datasets


def load_and_prepare_data(num_samples: int):
    """Loads MEDEC data with 4-way game structure.

    Uses only Error Flag = 1 rows (notes with errors):
    - 25% vanilla_harmful: Error note → copy as-is (EASY)
    - 25% adversarial_harmful: Error note → modify/worsen error (HARD)
    - 25% vanilla_benign: Clean note → copy as-is (EASY)
    - 25% adversarial_benign: Clean note → inject error (HARD)

    This balances attacker task difficulty (50% easy, 50% hard).
    
    Args:
        num_samples: Total number of samples (split 25/25/25/25)
        
    Returns:
        (ds_seeds, ds_few_shot) tuple:
            - ds_seeds: Dataset with 4-way game categories
            - ds_few_shot: Dataset with few-shot examples
    """
    print(
        "Loading MEDEC data with 4-way game structure (25/25/25/25)..."
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

    # Split samples into 4 equal groups (25% each)
    quarter_samples = num_samples // 4

    # Shuffle once for random distribution
    df_shuffled = df_errors.shuffle(seed=42).reset_index(drop=True)

    # Create 4 game categories
    datasets = []
    
    # 1. vanilla_harmful: Error note → copy as-is
    df_vanilla_harmful = df_shuffled.head(quarter_samples).copy()
    df_vanilla_harmful["game_category"] = "vanilla_harmful"
    df_vanilla_harmful["error_note"] = df_vanilla_harmful["Text"].str.strip()
    df_vanilla_harmful["clean_note"] = df_vanilla_harmful["Corrected Text"].str.strip()
    df_vanilla_harmful["error_type"] = df_vanilla_harmful["Error Type"]
    datasets.append(df_vanilla_harmful)
    
    # 2. adversarial_harmful: Error note → modify/worsen
    df_adv_harmful = df_shuffled.iloc[quarter_samples:quarter_samples*2].copy()
    df_adv_harmful["game_category"] = "adversarial_harmful"
    df_adv_harmful["error_note"] = df_adv_harmful["Text"].str.strip()
    df_adv_harmful["clean_note"] = df_adv_harmful["Corrected Text"].str.strip()
    df_adv_harmful["error_type"] = df_adv_harmful["Error Type"]
    datasets.append(df_adv_harmful)
    
    # 3. vanilla_benign: Clean note → copy as-is
    df_vanilla_benign = df_shuffled.iloc[quarter_samples*2:quarter_samples*3].copy()
    df_vanilla_benign["game_category"] = "vanilla_benign"
    df_vanilla_benign["error_note"] = df_vanilla_benign["Text"].str.strip()
    df_vanilla_benign["clean_note"] = df_vanilla_benign["Corrected Text"].str.strip()
    df_vanilla_benign["error_type"] = "none"  # No error to inject
    datasets.append(df_vanilla_benign)
    
    # 4. adversarial_benign: Clean note → inject error
    df_adv_benign = df_shuffled.iloc[quarter_samples*3:quarter_samples*4].copy()
    df_adv_benign["game_category"] = "adversarial_benign"
    df_adv_benign["error_note"] = df_adv_benign["Text"].str.strip()
    df_adv_benign["clean_note"] = df_adv_benign["Corrected Text"].str.strip()
    df_adv_benign["error_type"] = df_adv_benign["Error Type"]
    datasets.append(df_adv_benign)
    
    # Combine all 4 categories
    df_combined = pd.concat(datasets, ignore_index=True)
    
    # Create dataset with required columns
    ds_seeds = Dataset.from_pandas(
        df_combined[["game_category", "error_note", "clean_note", "error_type"]]
    ).shuffle(seed=44)

    # Few-shot examples: Show clean → error and error → modified_error
    df_few_shot = df_errors.head(5).copy()
    df_few_shot["clean_note"] = df_few_shot["Corrected Text"].str.strip()
    df_few_shot["error_note"] = df_few_shot["Text"].str.strip()
    df_few_shot["error_type"] = df_few_shot["Error Type"]

    ds_few_shot = Dataset.from_pandas(
        df_few_shot[["clean_note", "error_note", "error_type"]]
    )

    # Log distribution
    category_counts = df_combined["game_category"].value_counts()
    print(f"✅ Created 4-way game structure:")
    for category in ["vanilla_harmful", "adversarial_harmful", "vanilla_benign", "adversarial_benign"]:
        count = category_counts.get(category, 0)
        print(f"   - {category}: {count} ({100*count/len(df_combined):.1f}%)")
    print(f"✅ Few-shot examples: {len(ds_few_shot)}")
    
    return ds_seeds, ds_few_shot
