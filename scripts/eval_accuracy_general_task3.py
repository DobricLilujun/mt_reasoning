import pandas as pd
import numpy as np
import glob
import os
import pandas as pd

base_dir = "data/extraction_pdf/datasets/output_gammar_results"
all_jsonl = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True) 
task_files = [p for p in all_jsonl if "task3" in os.path.basename(p) or "task3" in p]  

df_summary = pd.DataFrame()
for f in task_files:
    print("Calculating for file:", f)
    df = pd.read_json(f, lines=True)

    total_score = 0 
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
        
        correct_grammar_letters = row["correct_grammars_letter"]
        if "gpt" not in f:
            log_prob_dict = row["log_probs"]["content"]
            ## Calculate NLL with Log Probabilities
            prob_accu = 0.0

            for correct_grammar_letter in correct_grammar_letters:
                # Find all log prob outputs
                for item in log_prob_dict:
                    if item['token'] == str(correct_grammar_letter):
                        logprob_list = item["top_logprobs"]
                        # Find the log prob for the correct grammar letter
                        for log_prob in logprob_list:
                            if str(correct_grammar_letter) == log_prob["token"]:
                                log_probs = log_prob["logprob"]
                                nll = -log_probs
                                probs_from_logprobs = np.exp(log_probs)
                                # print(f"Log probabilities for token '{correct_grammar_letter}': {log_probs}")
                                # print(f"Probabilities for token '{correct_grammar_letter}': {probs_from_logprobs}")

                                prob_accu += probs_from_logprobs
                                if np.isnan(nll):
                                    print(f"NaN found in NLL for row {idx}")
                                break
                        break
            row["prob_accu"] = prob_accu/len(correct_grammar_letters) if correct_grammar_letters else 2
        else:
            row["prob_accu"] = 0.0
        ## Calculate Accuracy
        correct_set = set(row['correct_grammars_letter'])
        try:
            detected_set = set(row["task3_dict"].get('grammar_selected', ''))
        except TypeError as e:
            if "unhashable type: 'dict'" in str(e):
                continue
            else:
                raise  
        
        score = len(detected_set & correct_set) / len(correct_set)
        total_score += score

        row["total_score"] = score
        row_df = pd.DataFrame([row])
        df_final = pd.concat([df_final, row_df], ignore_index=True)

    acc = total_score / len(df_final)
    std = np.sqrt(acc * (1 - acc) / len(df_final))

    probs_list = df_final["prob_accu"].dropna().values
    probs_mean = np.mean(probs_list)
    probs_std = np.std(probs_list)

    print(f"Accuracy: {total_score}/{len(df_final)} = {acc:.2f} ± {std:.2f}")
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

df_summary.to_json("data/extraction_pdf/datasets/SUMMARY/results_summary_task3.json", orient="records", lines=True)