from abc import ABC, abstractmethod
from src.llms.LLM_Wrappers import AbstractLLM
from tqdm import tqdm
import json, copy, os
from time import time
from datetime import datetime

class AbstractTAPipeline(ABC):
    def __init__(
        self,
        llm: AbstractLLM,
        input_path: str,
        output_dir: str | None = None,
        output_name: str | None = None,
        use_cache: bool = True
    ):
        self.llm = llm
        self.input_path = input_path
        self.output_dir = output_dir
        self.use_cache = use_cache

        self.output_path = self._make_output_path(input_path, output_dir, output_name)

        self.data = None
        self.codebook = None

    # ---------- Cache Utilities ----------

    def _get_cache_path(self) -> str:
        """Locate cache file one level above src/, named .ta_pipeline_cache.json."""
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        return os.path.join(project_root, ".ta_pipeline_cache.json")

    def _load_cache(self) -> dict:
        if not os.path.exists(self.cache_path):
            with open(self.cache_path, "w") as f:
                json.dump({}, f)
            return {}
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self, cache: dict):
        with open(self.cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    def _get_pipeline_name(self) -> str:
        """Use __str__ if implemented, otherwise default to class name."""
        try:
            name = str(self)
            if name.strip() == "":
                name = self.__class__.__name__
        except Exception:
            name = self.__class__.__name__
        return name

    def _check_cache(self) -> str | None:
        """Return cached output path if this model/input/pipeline combo was already processed."""
        if not self.use_cache:
            return None

        cache = self._load_cache()
        input_file = os.path.basename(self.input_path)
        model_name = self.llm.model_name
        pipeline_name = self._get_pipeline_name()

        try:
            cached_entry = cache[input_file][model_name][pipeline_name]
            cached_path = cached_entry["output_path"]
            if os.path.exists(cached_path):
                print(f"✅ Cached result found for {pipeline_name} ({model_name} on {input_file}).")
                return cached_path
        except KeyError:
            return None
        return None

    def _update_cache(self):
        """Update cache after a successful run."""
        cache = self._load_cache()
        input_file = os.path.basename(self.input_path)
        model_name = self.llm.model_name
        pipeline_name = self._get_pipeline_name()

        cache.setdefault(input_file, {})
        cache[input_file].setdefault(model_name, {})
        cache[input_file][model_name][pipeline_name] = {
            "output_path": self.output_path,
            "timestamp": datetime.now().isoformat()
        }
        self._save_cache(cache)

    # ---------- Path + Data Handling ----------

    def _make_output_path(self, input_path, output_dir=None, output_name=None):
        base_name = os.path.basename(input_path)
        base, ext = os.path.splitext(base_name)

        if output_name:
            annotated_name = f"{base}_{output_name}_annotated{ext}"
        else:
            annotated_name = f"{base}_annotated{ext}"
        output_name = annotated_name

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            return os.path.join(output_dir, annotated_name)
        else:
            dir_name = os.path.dirname(input_path)
            return os.path.join(dir_name, annotated_name)

    def load_data(self):
        with open(self.input_path, "r") as f:
            self.data = json.load(f)
        self.codebook = self.data["themes"]

    def save_data(self):
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _get_question_from_data(self) -> str:
        """Extracts the question text from the input data."""
        if self.data and "question" in self.data:
            return self.data["question"]
        print("⚠️ Warning: No question found in data.")
        return ""

    # ---------- Validation ----------

    @abstractmethod
    def annotate_entry(self, entry: dict) -> dict:
        pass

    def validate_output(self, entry: dict):
        if "annotations" not in entry:
            raise ValueError("Missing 'annotations' field after annotation.")
        return True

    def validate_annotation_structure(self, annotations: dict) -> bool:
        if not isinstance(annotations, dict):
            return False
        for theme, codes in annotations.items():
            if not isinstance(codes, dict):
                return False
            for code, details in codes.items():
                if not isinstance(details, dict):
                    return False
                if not {"section", "confidence", "annotator"}.issubset(details.keys()):
                    return False
                if not isinstance(details["section"], str):
                    return False
                if not isinstance(details["confidence"], (int, float)):
                    return False
                if not isinstance(details["annotator"], str):
                    return False
        return True

    # ---------- Main Run Logic ----------

    def run(self) -> str:
        """Runs the annotation pipeline with a live progress bar and caching support."""
        # Check cache before running
        self.cache_path = self._get_cache_path()
        cached_path = self._check_cache()
        if cached_path:
            print(f"Skipping run — returning cached file: {cached_path}")
            return cached_path

        self.load_data()
        annotated = copy.deepcopy(self.data)
        entries = annotated["answers"]
        total = len(entries)
        processed = 0

        with tqdm(total=total, desc="Annotating entries", unit="entry", ncols=90) as pbar:
            for entry in entries:
                text = entry.get("text", "").strip()
                if text == "":
                    entry = self.annotate_entry(entry)
                    self.validate_output(entry)
                    processed += 1
                    pbar.update(1)
                    continue

                start_time = time()
                entry = self.annotate_entry(entry)
                self.validate_output(entry)
                elapsed = time() - start_time

                processed += 1
                pbar.set_postfix_str(f"Last: {elapsed:.2f}s")
                pbar.update(1)

        self.data = annotated
        self.save_data()

        # Update cache after successful run
        self._update_cache()

        print(f"Annotated JSON written to {self.output_path}")
        return self.output_path
