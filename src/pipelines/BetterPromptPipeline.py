import json, os
from datetime import datetime
from src.llms.LLM_Wrappers import AbstractLLM
from src.pipelines.AbstractTAPipeline import AbstractTAPipeline
from src.pipelines.SimplePromptPipeline import SimplePromptPipeline

class BetterPromptDescPipeline(SimplePromptPipeline):
    def __str__(self):
        return "BetterPromptDescPipeline"

    def _format_codebook(self) -> dict:
        """
        Convert {theme: {code: description}} → {theme: [{"code":..., "description":...}, ...]}
        """
        formatted = {}
        for theme, codes in self.codebook.items():
            if isinstance(codes, dict):
                formatted[theme] = [
                    {"code": code, "description": desc or ""}
                    for code, desc in codes.items()
                ]
            else:
                formatted[theme] = [{"code": c, "description": ""} for c in codes]
        return formatted

    def annotate_entry(self, entry: dict) -> dict:
        text = entry.get("text", "").strip()

        # --- Handle blank text ----
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

        # --- Prepare codebook and question ----
        codebook_for_prompt = self._format_codebook()
        question_str = self._get_question_from_data()

        # --- Improved Prompt ---
        prompt = f"""
You are a highly accurate thematic annotator. You will receive a survey question, a response, 
and a detailed codebook. Your job is to determine which themes and codes apply to the response.
You must follow all rules exactly and output ONLY valid JSON.

=====================
INSTRUCTIONS
=====================
1. Use ONLY themes and codes exactly as listed in the codebook.
2. For each theme, evaluate all of its codes independently.
3. Each code includes a "description". Treat this description as authoritative:
   - Apply a code ONLY when the response clearly satisfies the description.
   - If the description implies exclusion conditions, obey them strictly.
4. Prefer precision over recall: do NOT guess or speculate.
5. Decide whether each code applies to:
   - the ENTIRE text → "section": ""
   - a SPECIFIC part → "section": "[start:end]"
6. Use Python-style character indices (0-based) on the original response.
7. Confidence must be a float between 0 and 1.
8. If no codes apply, return: {{"annotations": {{}}}}
9. Think carefully, but DO NOT reveal your reasoning. Output only the JSON object.
10. Do NOT add markdown, comments, or extra text.

=====================
OUTPUT SCHEMA (follow exactly)
=====================
{{
  "annotations": {{
    "<theme-name>": {{
      "<code-name>": {{
        "section": "[start:end]" or "",
        "confidence": 0.0-1.0,
        "annotator": "{self.llm.model_name}"
      }}
    }}
  }}
}}

=====================
QUESTION
=====================
{question_str}

=====================
CODEBOOK
=====================
{json.dumps(codebook_for_prompt, indent=2)}

=====================
RESPONSE TEXT
=====================
{json.dumps(text)}

Return ONLY the JSON object.
"""

        # --- Generate LLM output ---
        response = self.llm.generate(prompt)

        # --- Parse + Validate JSON ---
        try:
            result = self.llm.clean_and_parse_json(response)
            annotation = result.get("annotations", {})

            if self.validate_annotation_structure(annotation):
                entry["annotations"] = annotation
                self.log(f"Entry {entry['id']}: JSON processed successfully (better desc mode).")
            else:
                self.log(f"Entry {entry['id']}: Invalid JSON schema (better desc mode).")
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
            self.log(f"Entry {entry['id']}: JSON parsing error (better desc mode): {e}")
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

