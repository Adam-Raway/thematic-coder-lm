import json, os
from datetime import datetime
from src.llms.LLM_Wrappers import AbstractLLM
from src.pipelines.AbstractTAPipeline import AbstractTAPipeline

class SimplePromptPipeline(AbstractTAPipeline):
    def __init__(
        self,
        llm: AbstractLLM,
        input_path: str,
        output_dir: str | None = None,
        output_name: str | None = None,
        log_dir: str = "logs",
        use_cache: bool = True
    ):
        super().__init__(llm, input_path, output_dir, output_name, use_cache)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Single log file per pipeline run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        self.log_path = os.path.join(log_dir, f"{input_name}_{timestamp}.log")
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_file.write(f"[{ts}] {message}\n")
        self.log_file.flush()

    def annotate_entry(self, entry: dict) -> dict:
        text = entry.get("text", "").strip()

        # 1. Handle blank text entries automatically
        if not text:
            entry["annotations"] = {
                "No Responses": {
                    "Blank": {
                        "section": "",
                        "confidence": 1.0,
                        "annotator": self.llm.model_name
                    }
                }
            }
            self.log(f"Entry {entry['id']}: Blank text — annotated with 'Blank' code.")
            return entry

        # 2. Construct safe prompt
        prompt = f"""
        You are a thematic annotator. Based on the following text and codebook,
        return only a JSON object in the specified format (no explanations).

        Text: {json.dumps(text)}
        Codebook: {json.dumps(self.codebook, indent=2)}

        Output format:
        {{
          "annotations": {{
            "theme_name": {{
              "code_name": {{"section": "[start:end]", "confidence": float, "annotator": "{self.llm.model_name}"}}
            }}
          }}
        }}
        """

        # 3. Generate and parse JSON
        response = self.llm.generate(prompt)

        try:
            result = self.llm.clean_and_parse_json(response)
            annotation = result.get("annotations", {})

            if self.validate_annotation_structure(annotation):
                entry["annotations"] = annotation
                self.log(f"Entry {entry['id']}: JSON processed successfully.")
            else:
                self.log(f"Entry {entry['id']}: JSON produced but invalid format.")
                self.log(f"Raw LLM output:\n{response}\n{'-'*60}")
                entry["annotations"] = {
                    "Error": {
                        "InvalidFormat": {
                            "section": "",
                            "confidence": 0.0,
                            "annotator": self.llm.model_name
                        }
                    }
                }

        except Exception as e:
            self.log(f"Entry {entry['id']}: JSON parsing error: {e}")
            self.log(f"Raw LLM output:\n{response}\n{'-'*60}")
            entry["annotations"] = {
                "Error": {
                    "InvalidJSON": {
                        "section": "",
                        "confidence": 0.0,
                        "annotator": self.llm.model_name
                    }
                }
            }

        return entry

    def run(self) -> str:
        self.log(f"=== Pipeline started for {self.input_path} using {self.llm.model_name} ===")
        try:
            output_path = super().run()
            self.log(f"Pipeline completed successfully. Output at {output_path}")
        except Exception as e:
            self.log(f"Pipeline failed: {e}")
            raise
        finally:
            self.log_file.close()
            print(f"Logs saved to {self.log_path}")
        return output_path


if __name__ == "__main__":
    llm = AbstractLLM.from_name("qwen3:4b")
    pipeline = SimplePromptPipeline(
        llm,
        "src/data/test.json",
        output_dir="outputs/",
        output_name="qwen3:4b",
        use_cache=True
    )
    pipeline.run()
