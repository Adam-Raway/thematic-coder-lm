import json
from collections import defaultdict

class Evaluator:
    """
    Evaluates an auto-annotated JSON file against a ground truth annotated JSON file.
    Computes precision and recall globally, per theme, and per (theme, code).
    """

    def __init__(self, auto_path: str, gt_path: str):
        self.auto_path = auto_path
        self.gt_path = gt_path

        with open(auto_path, "r") as f:
            self.auto_data = json.load(f)
        with open(gt_path, "r") as f:
            self.gt_data = json.load(f)

        # Validate structure
        self._validate_structure()

    def _validate_structure(self):
        """Checks that both JSONs are compatible and that texts match."""
        if self.auto_data["question"] != self.gt_data["question"]:
            print("⚠️ Warning: Questions differ between files.")

        auto_answers = self.auto_data["answers"]
        gt_answers = self.gt_data["answers"]

        if len(auto_answers) != len(gt_answers):
            raise ValueError("Number of answers differ between auto and ground truth files.")

        for a_entry, g_entry in zip(auto_answers, gt_answers):
            if a_entry["id"] != g_entry["id"]:
                raise ValueError(f"Answer ID mismatch: {a_entry['id']} vs {g_entry['id']}")
            if a_entry["text"].strip() != g_entry["text"].strip():
                print(f"⚠️ Warning: Text mismatch for ID {a_entry['id']}")

    def _collect_codes(self, annotations: dict, min_confidence: float) -> set:
        """
        Flattens annotations into a set of (theme, code) pairs
        meeting the min_confidence threshold (inclusive).
        """
        pairs = set()
        for theme, codes in annotations.items():
            for code, details in codes.items():
                confidence = details.get("confidence", 1.0)
                if confidence >= min_confidence:
                    pairs.add((theme, code))
        return pairs

    def evaluate_precision_recall(self, min_confidence: float = 0.5) -> dict:
        """
        Evaluates precision, recall, and f1-score globally and per theme/code.
        Returns a dictionary of metrics.
        """
        tp_global, fp_global, fn_global = 0, 0, 0
        per_theme_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        per_code_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for auto_entry, gt_entry in zip(self.auto_data["answers"], self.gt_data["answers"]):
            auto_codes = self._collect_codes(auto_entry.get("annotations", {}), min_confidence)
            gt_codes = self._collect_codes(gt_entry.get("annotations", {}), min_confidence)

            # Compute TP, FP, FN
            tps = auto_codes & gt_codes
            fps = auto_codes - gt_codes
            fns = gt_codes - auto_codes

            tp_global += len(tps)
            fp_global += len(fps)
            fn_global += len(fns)

            # Per theme/code tracking
            for theme, code in tps:
                per_theme_counts[theme]["tp"] += 1
                per_code_counts[(theme, code)]["tp"] += 1
            for theme, code in fps:
                per_theme_counts[theme]["fp"] += 1
                per_code_counts[(theme, code)]["fp"] += 1
            for theme, code in fns:
                per_theme_counts[theme]["fn"] += 1
                per_code_counts[(theme, code)]["fn"] += 1

        # Compute metrics
        def safe_div(num, denom):
            return num / denom if denom > 0 else 0.0

        results = {
            "global": {
                "precision": safe_div(tp_global, tp_global + fp_global),
                "recall": safe_div(tp_global, tp_global + fn_global),
                "f1-score": safe_div(2 * tp_global, 2 * tp_global + fp_global + fn_global)
            },
            "per_theme": {},
            "per_code": {},
        }

        for theme, counts in per_theme_counts.items():
            results["per_theme"][theme] = {
                "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
                "f1-score": safe_div(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
            }

        for (theme, code), counts in per_code_counts.items():
            key = f"{theme}|{code}"  # string key for JSON compatibility
            results["per_code"][key] = {
                "precision": safe_div(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": safe_div(counts["tp"], counts["tp"] + counts["fn"]),
                "f1-score": safe_div(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
            }


        return results

if __name__ == "__main__":
    # Example usage
    evaluator = Evaluator(auto_path="outputs/Q17_Annotated_Responses_annotated.json", gt_path="src/data/Q17_Annotated_Responses.json")
    metrics = evaluator.evaluate_precision_recall(min_confidence=0.7)
    print(json.dumps(metrics, indent=2))