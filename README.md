# thematic-coder-lm
A research project for testing the effectiveness of different LLM pipelines at performing Thematic Coding on education-related data.

## Setting up the Project

### Requirements

- You must have Ollama downloaded in order to run the Qwen3 models used in these pipelines. If you have downloaded Ollama already, you can download the Qwen3 models using [download_local_models.py](/src/llms/download_local_models.py).

### Creating your datasets

To run the program using your own text data, store the text as JSON files in [/src/data](/src/data).

QnA data must follow the following format, where each text has annotations in a hierarchical structure with themes that contain codes. Thus, an unannotated text will simply have `"annotations" = {}`. "section" describes the part of the text to which the code applies to, and follows mostly the same conventions as Python string indexing. If the code applies to the entire text, section can just be an empty string.
```json
{
  "question": "What is ...",
  "themes" : {
    "<theme-name>": ["<code-name>", "..."]
  },
  "answers" : [
    {
      "id": 0,
      "text": "...",
      "annotations": {
        "<theme-name>": {
          "<code-name>": {"section": "", "confidence": 1, "annotator": "human"},
          "<code-name>": {"section": "[0: 5]", "confidence": 1, "annotator": "human"}
        },
        "<theme-name>": {
          "<code-name>": {"section": "[0: 5]", "confidence": 0.9, "annotator": "gpt-4"}
        }
      }
    },
    {
      "id": 1,
      "text": "...",
      "annotations": {}
    }
  ]
}


```
