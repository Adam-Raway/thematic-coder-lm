import json, os
from datetime import datetime
from src.llms.LLM_Wrappers import AbstractLLM
from src.pipelines.AbstractTAPipeline import AbstractTAPipeline


class BetterPromptPipeline(AbstractTAPipeline):
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        self.log_path = os.path.join(log_dir, f"{input_name}_{timestamp}.log")
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def __str__(self):
        return "BetterPromptPipeline"

    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_file.write(f"[{ts}] {message}\n")
        self.log_file.flush()

    def _format_codebook(self) -> dict:
        """
        Converts the new format {theme: {code: desc}} into
        {theme: [code, ...]} for prompt compatibility.
        """
        formatted = {}
        for theme, codes in self.codebook.items():
            # ignore descriptions
            formatted[theme] = list(codes.keys()) if isinstance(codes, dict) else codes
        return formatted

    def annotate_entry(self, entry: dict) -> dict:
        text = entry.get("text", "").strip()

        # 1. Handle blank text
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

        # 2. Format codebook (ignore descriptions)
        codebook_for_prompt = self._format_codebook()

        # 3. Construct prompt
        prompt = f"""
        You are a highly accurate thematic annotator. Given a survey question and its response, you apply qualitative codes strictly using the provided codebook to the response. Follow all instructions precisely and output *only* valid JSON.

        INSTRUCTIONS:
        1. Use only themes and codes that appear in the codebook. Never invent new codes.
        2. Apply a code only if the text clearly supports it. Avoid speculative inference.
        3. If a code applies to the entire text, set "section": "".
        4. If a code applies to part of the text, use Python-style character index slicing: "[start:end]".
        5. Confidence must be a float between 0 and 1.
        6. If no codes apply, return: {{"annotations": {{}}}}
        7. Think step-by-step internally, but output only the final JSON object.
        8. Output strictly valid JSON — no explanations, no notes, no markdown, no code fences.
        9. Include only themes that contain at least one detected code.

        OUTPUT SCHEMA (follow exactly):
        {{
        "annotations": {{
            "<theme-name>": {{
            "<code-name>": {{
                "section": "[start:end]",
                "confidence": float,
                "annotator": "{self.llm.model_name}"
            }}
            }}
        }}
        }}

        QUESTION:
        {self._get_question_from_data()}

        CODEBOOK:
        {json.dumps(codebook_for_prompt, indent=2)}

        TEXT:
        {json.dumps(text)}

        Return ONLY the JSON object.
        """

        # 4. Generate + parse JSON
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



# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    llm = AbstractLLM.from_name("gpt-4o-mini")

    # Normal (ignores descriptions)
    pipeline = BetterPromptPipeline(
        llm,
        "src/data/test.json",
        output_dir="outputs/",
        output_name="gpt-4o-mini",
        use_cache=False
    )
    pipeline.run()

