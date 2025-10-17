import pandas as pd
import numpy as np
import glob
import os
import pandas as pd

base_dir = "data/extraction_pdf/datasets/output_gammar_results"
all_jsonl = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True) 
task_files = [p for p in all_jsonl if "task1" in os.path.basename(p) or "task1" in p]  

df_summary = pd.DataFrame()
for f in task_files:
    print("Calculating for file:", f)
    # if "20251004_210838" not in f:
    #     continue
    df = pd.read_json(f, lines=True)
    num_correct = 0
    total = len(df)
    df_final = pd.DataFrame()
    for idx, row in df.iterrows():

        key = f.split("/")[-1].split("_")[0]+"_dict"
        if key =="task4_dict":
            key = "output_dict_mp"
        if pd.isnull(row[key]):
            print(f"Warning: Missing {key} in row {idx}.")
            continue

        # print(f"Processing row {idx}")
        
        correct_grammar_letter = row["correct_grammar_letter"]
        ## Calculate NLL with Log Probabilities
        if "gpt" not in f:
            log_prob_dict = row["log_probs"]["content"]
            for item in log_prob_dict:
                # print(f"Decoded token: {item['token']}, Number: {item['bytes']}")
                if item['token'] == correct_grammar_letter:
                    logprob_list = item["top_logprobs"]
                    for log_prob_dict in logprob_list:
                        if correct_grammar_letter == log_prob_dict["token"]:
                            log_probs = log_prob_dict["logprob"]
                            nll = -log_probs
                            probs_from_logprobs = np.exp(log_probs)
                            # print(f"Log probabilities for token '{correct_grammar_letter}': {log_probs}")
                            # print(f"Probabilities for token '{correct_grammar_letter}': {probs_from_logprobs}")
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
        correct_grammar_letter = row['correct_grammar_letter']
        detected_grammar_letter = str(row['task1_dict'].get('grammar_selected') or "").strip().upper()
        
        
        if correct_grammar_letter == detected_grammar_letter:
            num_correct += 1
            row['is_correct'] = True
        
        else:
            row['is_correct'] = False
        
        row_df = pd.DataFrame([row])
        df_final = pd.concat([df_final, row_df], ignore_index=True)

    acc = num_correct / len(df_final)
    std = np.sqrt(acc * (1 - acc) /  len(df_final))
    probs_list = df_final["probs_from_logprobs"].dropna().values
    probs_mean = np.mean(probs_list)
    probs_std = np.std(probs_list)

    print(f"Accuracy: {num_correct}/{ len(df_final)} = {acc:.2f} ± {std:.2f}")
    print(f"Probabilities: {probs_mean:.2f} ± {probs_std:.2f}")

    str_accuracy = f"{acc:.2f}_±_{std:.2f}"
    str_probs = f"{probs_mean:.2f}_±_{probs_std:.2f}"
    row = {
        "file_name": os.path.basename(f),
        "str_accuracy": str_accuracy,
        "str_probs": str_probs,
        "accuracy": acc,
        "accuracy_std": std,
        "probs_mean": probs_mean,
        "probs_std": probs_std 

    }
    row_df = pd.DataFrame([row])
    df_summary = pd.concat([df_summary, row_df], ignore_index=True)

df_summary.to_json("data/extraction_pdf/datasets/SUMMARY/results_summary_task1.json", orient="records", lines=True)