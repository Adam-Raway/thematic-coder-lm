import json
import os
import copy
from tqdm import tqdm
from time import time
from datetime import datetime
from src.llms.LLM_Wrappers import AbstractLLM
from src.pipelines.SimplePromptPipeline import SimplePromptPipeline # Assuming SimplePromptPipeline is imported from a relevant path

class FewShotPipeline(SimplePromptPipeline):
    def __init__(
        self,
        llm: AbstractLLM,
        input_path: str,
        example_ids: list[int],
        output_dir: str | None = None,
        output_name: str | None = None,
        log_dir: str = "logs",
        use_cache: bool = True
    ):
        # We must change the default output name to reflect the partial save
        if output_name is None:
             output_name = "partial_few_shot"
        
        super().__init__(llm, input_path, output_dir, output_name, log_dir, use_cache)
        self.example_ids = example_ids
        self.examples_context = "" 
        self.llm_annotator_tag = f"{self.llm.model_name}_llm" # Updated annotator tag

    def __str__(self):
        return "FewShotPipeline"

    def _build_examples_context(self):
        """
        Retrieves the text and existing annotations for the provided example_ids
        and formats them into a string for the prompt.
        """
        if not self.data:
            self.load_data()

        entry_map = {entry['id']: entry for entry in self.data['answers']}
        examples_list = []
        
        for ex_id in self.example_ids:
            entry = entry_map.get(ex_id)
            if entry is None:
                self.log(f"⚠️ Warning: Example ID {ex_id} not found in data. Skipping.")
                continue
            
            # Use original annotations for examples
            annotations = entry.get("annotations")
            if not annotations:
                self.log(f"⚠️ Warning: Example ID {ex_id} has no annotations. Skipping.")
                continue

            # Format the example for the prompt (Note: we serialize the whole 'annotations' block)
            example_str = (
                f"Example Input: {json.dumps(entry['text'])}\n"
                f"Example Output: {json.dumps({'annotations': annotations})}"
            )
            examples_list.append(example_str)

        if not examples_list:
            self.log("❌ Error: No valid examples found. Pipeline will behave like Zero-Shot.")
            return ""

        return "\n\n---\n\n".join(examples_list)

    def annotate_entry(self, entry: dict) -> dict:
        """
        Overriding the annotation logic to include examples and updated instructions.
        """
        text = entry.get("text", "").strip()
        
        # 1. Handle blank text
        if not text:
            # Preserve existing human annotation structure if possible, but update annotator
            entry["annotations"] = {
                "No Responses": {
                    "Blank": {
                        "section": "",
                        "confidence": 1.0, # Retain 1.0 for definitive blank
                        "annotator": "human" # Assuming Blank/NoRelevant are usually pre-labeled
                    }
                }
            }
            return entry

        # 2. Lazy load examples context
        if not self.examples_context:
            self.examples_context = self._build_examples_context()

        # 3. Format codebook
        codebook_for_prompt = self._format_codebook()

        # 4. Construct Few-Shot Prompt with updated confidence instruction
        prompt = f"""
        You are a thematic annotator. I will provide you with a Codebook and several labeled Examples. 
        Your task is to annotate the "Target Text" following the patterns shown in the examples.
        
        Return **only** a JSON object (no markdown, no explanations).

        **Crucially, you must assign a `confidence` rating as a float between 0.0 (low) and 1.0 (high) based on how certain you are of the annotation.**
        
        === CODEBOOK ===
        {json.dumps(codebook_for_prompt, indent=2)}

        === EXAMPLES ===
        {self.examples_context}

        === TARGET TEXT ===
        Input: {json.dumps(text)}

        Output format:
        {{
          "annotations": {{
            "theme_name": {{
              "code_name": {{"section": "[substring]", "confidence": float, "annotator": "{self.llm_annotator_tag}"}}
            }}
          }}
        }}
        """

        # 5. Generate + parse JSON (using parent's logging/parsing logic)
        response = self.llm.generate(prompt)

        try:
            result = self.llm.clean_and_parse_json(response)
            annotation = result.get("annotations", {})

            # Update annotator field for all generated codes (Tweak 2)
            for theme, codes in annotation.items():
                for code, details in codes.items():
                    # Check if 'annotator' is missing or needs updating (Tweak 2)
                    if details.get("annotator") != self.llm_annotator_tag:
                         details["annotator"] = self.llm_annotator_tag
                    
                    # Ensure confidence is a float (Tweak 3)
                    if "confidence" in details:
                        details["confidence"] = float(details["confidence"])


            if self.validate_annotation_structure(annotation):
                entry["annotations"] = annotation
                self.log(f"Entry {entry['id']}: JSON processed successfully.")
            else:
                self.log(f"Entry {entry['id']}: Invalid format received.")
                entry["annotations"] = {
                    "Error": {
                        "InvalidFormat": {
                            "section": "",
                            "confidence": 0.0,
                            "annotator": self.llm_annotator_tag
                        }
                    }
                }

        except Exception as e:
            self.log(f"Entry {entry['id']}: JSON parsing error: {e}")
            entry["annotations"] = {
                "Error": {
                    "InvalidJSON": {
                        "section": "",
                        "confidence": 0.0,
                        "annotator": self.llm_annotator_tag
                    }
                }
            }

        return entry

    def run_single(self, target_id: int) -> dict:
        """
        Runs the annotation pipeline on a specific entry ID only.
        Saves a new file containing ONLY the examples and the annotated target entry (Tweak 1).
        """
        self.log(f"=== Single Entry Run started for ID: {target_id} ===")
        
        if self.data is None:
            self.load_data()

        # Find the target entry and its original index
        target_entry = None
        target_index = -1
        
        for idx, entry in enumerate(self.data["answers"]):
            if entry["id"] == target_id:
                target_entry = entry
                target_index = idx
                break
        
        if target_entry is None:
            msg = f"❌ Error: Entry ID {target_id} not found in input file."
            print(msg)
            self.log(msg)
            return {}

        # 1. Annotate the target entry
        annotated_target_entry = self.annotate_entry(copy.deepcopy(target_entry))
        
        # 2. Collect entries for the new output file (Tweak 1)
        
        # Start with the original data structure (copy everything but answers)
        new_data = copy.deepcopy(self.data)
        new_data["answers"] = []
        
        # Map of all entries for easy lookup
        entry_map = {entry['id']: entry for entry in self.data['answers']}
        
        # Add all example entries (with their original human annotations)
        for ex_id in self.example_ids:
            if ex_id in entry_map:
                new_data["answers"].append(entry_map[ex_id])
        
        # Add the newly annotated target entry
        new_data["answers"].append(annotated_target_entry)
        
        # 3. Validation and Saving
        try:
            self.validate_output(annotated_target_entry)
        except ValueError as e:
            self.log(f"Validation failed: {e}")
            
        # Update the main data object for the final save
        self.data = new_data
        self.save_data()

        self.log(f"Entry {target_id} annotated and saved to {self.output_path}. File contains examples + target only.")
        print(f"✅ Entry {target_id} annotated and saved (partial file) to {self.output_path}")
        
        return annotated_target_entry
    
    def run_multiple(self, target_ids: list[int]) -> list[dict]:
        """
        Runs the annotation pipeline on a list of target entry IDs.
        Saves a new file containing ONLY the examples and the annotated target entries.
        """
        self.log(f"=== Multiple Entries Run started for IDs: {target_ids} ===")
        
        if self.data is None:
            self.load_data()

        # Map of all entries for easy lookup
        entry_map = {entry['id']: entry for entry in self.data['answers']}
        
        annotated_entries = []
        
        # Use tqdm for progress tracking
        with tqdm(total=len(target_ids), desc="Annotating entries", unit="entry", ncols=90) as pbar:
            for target_id in target_ids:
                start_time = time()
                
                target_entry = entry_map.get(target_id)
                
                if target_entry is None:
                    msg = f"❌ Error: Entry ID {target_id} not found in input file. Skipping."
                    self.log(msg)
                    print(msg)
                    pbar.update(1)
                    continue

                # 1. Annotate the target entry (using a deepcopy to avoid mutating source data)
                annotated_target_entry = self.annotate_entry(copy.deepcopy(target_entry))
                
                # 2. Validation
                try:
                    self.validate_output(annotated_target_entry)
                except ValueError as e:
                    self.log(f"Entry {target_id} Validation failed: {e}")
                
                annotated_entries.append(annotated_target_entry)
                
                elapsed = time() - start_time
                pbar.set_postfix_str(f"Last: {elapsed:.2f}s")
                pbar.update(1)

        # 3. Collect entries for the new output file (Selective Saving)
        
        new_data = copy.deepcopy(self.data)
        new_data["answers"] = []
        
        # # Add all example entries (with their original human annotations)
        # for ex_id in self.example_ids:
        #     if ex_id in entry_map:
        #         new_data["answers"].append(entry_map[ex_id])
        
        # Add the newly annotated target entries
        new_data["answers"].extend(annotated_entries)
        
        # 4. Save to disk
        self.data = new_data
        self.save_data()

        self.log(f"Annotated batch saved to {self.output_path}. File contains examples + targets only.")
        print(f"✅ Annotated batch of {len(target_ids)} entries saved (partial file) to {self.output_path}")
        
        return annotated_entries