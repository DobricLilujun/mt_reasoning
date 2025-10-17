import pandas as pd
import numpy as np
import glob
import os
import pandas as pd

base_dir = "data/extraction_pdf/datasets/output_gammar_results"
all_jsonl = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True) 
task_files = [p for p in all_jsonl if "task4" in os.path.basename(p) or "task4" in p]  
log_prob_name = "log_prob_mp"  # log_prob_mp_single_1  log_prob_mp_single_2 log_prob_mp_with_grammar_desc
output_dict_name = "output_dict_mp"  # output_dict_mp_single_1  output_dict_mp_single_2 output_dict_mp_with_grammar_desc
df_summary = pd.DataFrame()
for f in task_files:
    print("Calculating for file:", f)
    df = pd.read_json(f, lines=True)
    total_score = 0 
    num_correct = 0
    total = len(df)
    df_final = pd.DataFrame()
    for idx, row in df.iterrows():

        # print(f"Processing row {idx}")
        key = f.split("/")[-1].split("_")[0]+"_dict"
        if key =="task4_dict":
            key = "output_dict_mp"
        if pd.isnull(row[key]):
            print(f"Warning: Missing {key} in row {idx}.")
            continue

        # print(f"Processing row {idx}")
        
        correct_sentence_letter = row["correct_sentence_letter"]

        if "gpt" not in f:
            log_prob_dict = row[log_prob_name]["content"]
            ## Calculate NLL with Log Probabilities
            for item in log_prob_dict:
                # print(f"Decoded token: {item['token']}, Number: {item['bytes']}")
                if item['token'] == correct_sentence_letter:
                    logprob_list = item["top_logprobs"]
                    for log_prob_dict in logprob_list:
                        if correct_sentence_letter == log_prob_dict["token"]:
                            log_probs = log_prob_dict["logprob"]
                            nll = -log_probs
                            probs_from_logprobs = np.exp(log_probs)
                            row["log_probs_value"] = log_probs
                            row["nll"] = nll
                            row["probs_from_logprobs"] = probs_from_logprobs
                            # print(f"Probabilities for token '{correct_grammar_letter}': {probs_from_logprobs}")
                            if np.isnan(nll):
                                print(f"NaN found in NLL for row {idx}")
                            break
        else:
            row["log_probs_value"] = 0.0
            row["nll"] = 0.0
            row["probs_from_logprobs"] = 0.0

        ## Calculate Accuracy
        correct_sentence_letter = row['correct_sentence_letter']
        detected_sentence_letter = row[output_dict_name].get('sentence_selected', '').strip().upper()
        if correct_sentence_letter == detected_sentence_letter:
            num_correct += 1
            row['is_correct'] = True
        
        else:
            row['is_correct'] = False
        
        row_df = pd.DataFrame([row])
        df_final = pd.concat([df_final, row_df], ignore_index=True)

    acc = num_correct / len(df_final)
    std = np.sqrt(acc * (1 - acc) / len(df_final))
    probs_list = df_final["probs_from_logprobs"].dropna().values
    probs_mean = np.mean(probs_list)
    probs_std = np.std(probs_list)

    print(f"Accuracy: {num_correct}/{len(df_final)} = {acc:.2f} ± {std:.2f}")
    print(f"Probabilities: {probs_mean:.2f} ± {probs_std:.2f}")


    str_accuracy = f"{acc:.2f}_±_{std:.2f}"
    str_probs = f"{probs_mean:.2f}_±_{probs_std:.2f}"
    row = {
        "file_name": os.path.basename(f),
        "total_samples": len(df_final),
        "total_original_samples": len(df),
        "str_accuracy": str_accuracy,
        "str_probs": str_probs,
        "accuracy": acc,
        "accuracy_std": std,
        "probs_mean": probs_mean,
        "probs_std": probs_std 

    }
    row_df = pd.DataFrame([row])
    df_summary = pd.concat([df_summary, row_df], ignore_index=True)

df_summary.to_json("data/extraction_pdf/datasets/SUMMARY/results_summary_task4.json", orient="records", lines=True)
