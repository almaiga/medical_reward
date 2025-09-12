import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import concatenate_datasets, Dataset
import argparse
import os
from datetime import datetime
import re
from tqdm import tqdm

# --- Configuration ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_model_and_tokenizer(model_id):
    """Loads the specified model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
       # quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def load_medec_benchmark(num_ai_errors=60, num_doctor_errors=60):
    """
    Loads and prepares the MEDEC benchmark dataset from local CSV files,
    separating few-shot examples from the main evaluation set.
    """
    print("Loading MEDEC benchmark data from local CSV files...")
    try:
        path = "data_copy/MEDEC/"
        uw_df = pd.read_csv(path + "MEDEC-UW/MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv")
        ms_df = pd.read_csv(path + "MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv")
        print(uw_df.columns)
        # --- FIX: Drop rows with missing 'Error Flag' to ensure data quality ---
        uw_df.dropna(subset=['Error Flag'], inplace=True)
        ms_df.dropna(subset=['Error Flag'], inplace=True)

        # Convert columns to a nullable integer type to handle potential NaNs gracefully
        # and ensure both dataframes have the same type before creating datasets.
        uw_df['Error Flag'] = uw_df['Error Flag'].astype('Int64')
        ms_df['Error Flag'] = ms_df['Error Flag'].astype('Int64')
        uw_df['Error Sentence ID'] = uw_df['Error Sentence ID'].astype('Int64')
        ms_df['Error Sentence ID'] = ms_df['Error Sentence ID'].astype('Int64')

        print(f"Loaded {len(uw_df)} examples from MEDEC-UW and {len(ms_df)} from MEDEC-MS.")
        # Extract 1 positive and 1 negative example from each dataframe
        few_shot_dfs = [
            ms_df[ms_df['Error Flag'] == 1].head(1), ms_df[ms_df['Error Flag'] == 0].head(1),
            uw_df[uw_df['Error Flag'] == 1].head(1), uw_df[uw_df['Error Flag'] == 0].head(1)
        ]
        few_shot_examples = pd.concat(few_shot_dfs)
        print(f"Selected {len(few_shot_examples)} few-shot examples for the prompt.")
        # Exclude few-shot examples from the main dataframes
        ms_main_df = ms_df.drop(few_shot_examples.index.intersection(ms_df.index)).head(num_ai_errors)
        uw_main_df = uw_df.drop(few_shot_examples.index.intersection(uw_df.index)).head(num_doctor_errors)

        # Create the final benchmark dataset
        benchmark_dataset = concatenate_datasets([
            Dataset.from_pandas(ms_main_df), Dataset.from_pandas(uw_main_df)
        ])
        
        print(f"Loaded {len(benchmark_dataset)} examples for evaluation. Using {len(few_shot_examples)} few-shot examples.")
        return benchmark_dataset, few_shot_examples
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return None, None

def run_inference(model, tokenizer, dataset, few_shot_examples):
    """Runs inference on the dataset and collects model outputs."""
    results = []

    prompt_system = "You are a skilled medical doctor reviewing clinical texts for accuracy."
    prompt_user_template = (
        "Each text is either entirely correct or contains exactly one medical error.\n"
        "Medical errors are clinically harmful inaccuracies in patient records.\n"
        "Analyze the following medical note carefully.\n"
        "Medical Note: {medical_note}\n"
        "Respond in JSON format with two fields:\n"
        '- \"answer\": either \"CORRECT\" or \"INCORRECT\"\n'
        '- \"reasoning\": a brief, single-sentence explanation for your decision (no more than 25 words).\n'
    )

    # Create the few-shot prompt prefix once
    few_shot_examples_text = ""
    for _, example in few_shot_examples.iterrows():
        label = "INCORRECT" if example['Error Flag'] == 1 else "CORRECT"
        reasoning = "The note contains a factual error." if label == "INCORRECT" else "The note is factually correct."
        few_shot_examples_text += f"\n### Medical Note:\n{example['Text']}\n### Answer:\n{{\"answer\": \"{label}\", \"reasoning\": \"{reasoning}\"}}"

    for item in tqdm(dataset, desc="Inference"):
        medical_note = item['Text']
        error_flag = item['Error Flag']

        # Combine the few-shot prefix with the current example
        prompt_user = f"{prompt_user_template.format(medical_note=medical_note)}\n{few_shot_examples_text}"
        
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate the output
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, # Increased token count for reasoning
            temperature=0.1, 
            do_sample=False, # Disabled sampling for more deterministic output
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Parse the model's answer and reasoning
        answer_match = re.search(r'"answer"\s*:\s*"?(CORRECT|INCORRECT)"?', response_text, re.IGNORECASE)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"', response_text, re.DOTALL)
        
        model_label = answer_match.group(1).upper() if answer_match else "UNKNOWN"
        model_reasoning = reasoning_match.group(1).strip() if reasoning_match else "NO REASONING"
        gt_label = "INCORRECT" if error_flag == 1 else "CORRECT"

        results.append({
            'Text ID': item['Text ID'],
            'model_label': model_label,
            'model_reasoning': model_reasoning,
            'ground_truth_label': gt_label,
            'Error Flag': error_flag,
            'full_response': response_text
        })
        
    return pd.DataFrame(results)

def main():
    """Main function to run the baseline evaluation."""
    parser = argparse.ArgumentParser(description="Run baseline evaluation on a medical LLM.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        required=True, 
        help="The Hugging Face model ID to evaluate (e.g., 'Qwen/Qwen2-4B-Instruct')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The directory to save the CSV results file."
    )
    args = parser.parse_args()

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    benchmark_data, few_shot_examples = load_medec_benchmark()

    if benchmark_data is None:
        return

    # Run inference
    results_df = run_inference(model, tokenizer, benchmark_data, few_shot_examples)
    
    # --- Generate dynamic filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_id.replace("/", "_")
    num_samples = len(benchmark_data)
    output_filename = f"{timestamp}_{model_name_safe}_{num_samples}_samples.csv"
    
    # Create output directory and save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)
        
    results_df.to_csv(output_path, index=False)
    print(f"Baseline evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()