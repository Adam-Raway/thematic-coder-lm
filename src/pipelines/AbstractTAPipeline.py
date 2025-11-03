from abc import ABC, abstractmethod
from src.llms.LLM_Wrappers import AbstractLLM
from tqdm import tqdm
import json, copy, os

class AbstractTAPipeline(ABC):
    def __init__(self, llm: AbstractLLM, input_path: str, output_dir: str | None = None):
        self.llm = llm
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_path = self._make_output_path(input_path, output_dir)
        self.data = None
        self.codebook = None

    def _make_output_path(self, input_path, output_dir=None):
        base_name = os.path.basename(input_path)
        base, ext = os.path.splitext(base_name)

        # If output_dir is provided, save there
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            return os.path.join(output_dir, f"{base}_annotated{ext}")
        else:
            # Default: same folder as input file
            dir_name = os.path.dirname(input_path)
            return os.path.join(dir_name, f"{base}_annotated{ext}")

    def load_data(self):
        with open(self.input_path, "r") as f:
            self.data = json.load(f)
        self.codebook = self.data["themes"]

    def save_data(self):
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)

    @abstractmethod
    def annotate_entry(self, entry: dict) -> dict:
        pass

    def validate_output(self, entry: dict):
        if "annotations" not in entry:
            raise ValueError("Missing 'annotations' field after annotation.")
        return True

    def run(self):
        """Runs the annotation pipeline with a live progress bar."""
        self.load_data()
        annotated = copy.deepcopy(self.data)
        entries = annotated["answers"]
        total = len(entries)
        processed = 0

        # tqdm progress bar
        with tqdm(total=total, desc="Annotating entries", unit="entry", ncols=90) as pbar:
            for entry in entries:
                text = entry.get("text", "").strip()
                if text == "":
                    # Allow blank handling to occur in child class
                    entry = self.annotate_entry(entry)
                    self.validate_output(entry)
                    processed += 1
                    pbar.update(1)
                    continue

                start_time = time.time()
                entry = self.annotate_entry(entry)
                self.validate_output(entry)
                elapsed = time.time() - start_time

                processed += 1
                pbar.set_postfix_str(f"Last: {elapsed:.2f}s")
                pbar.update(1)

        self.data = annotated
        self.save_data()
        print(f"Annotated JSON written to {self.output_path}")
