from datasets import load_dataset
import torch

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("wenhu/Health-Bench")["test"]

# Filter rows with non-empty 'ideal_completions_data'
filtered_ds = ds.filter(lambda x: x["ideal_completions_data"] is not None and len(x["ideal_completions_data"]) > 0)

print(filtered_ds)
print(filtered_ds[0])

# for criterion in filtered_ds[0]["rubrics"]:
#     print(criterion)
#     print("-"*60)

