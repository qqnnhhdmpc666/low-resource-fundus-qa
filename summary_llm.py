# summary_llm.py
import json
import numpy as np

def main():
    try:
        # Load llm_scores.json
        with open("llm_scores.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Initialize scores dictionary
        scores = {k: [] for k in ["correctness", "completeness", "safety", "helpfulness"]}
        
        # Extract scores from each item
        for item in data["details"]:
            if "llm_score" in item:
                llm_score = item["llm_score"]
                for k in scores:
                    if k in llm_score:
                        scores[k].append(llm_score[k])
        
        # Calculate and print mean scores
        print("LLM Evaluation Summary:")
        for k in scores:
            if scores[k]:
                mean_score = np.mean(scores[k])
                print(f"{k}: {mean_score:.4f}")
            else:
                print(f"{k}: No scores available")
        
        # Add summary to the data and save
        summary_scores = {k: np.mean(v) if v else 0 for k, v in scores.items()}
        data["llm_summary"] = summary_scores
        
        with open("llm_scores_with_summary.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("\nSummary saved to llm_scores_with_summary.json")
        
    except FileNotFoundError:
        print("Error: llm_scores.json not found.")
    except json.JSONDecodeError:
        print("Error: llm_scores.json is not valid JSON.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
