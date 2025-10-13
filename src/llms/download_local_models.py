import subprocess

# Select the Qwen3 model to download.
model = "qwen3:4b"

def download_qwen_model():
    """
    Downloads the Qwen3 model using Ollama.
    """
    # Check Ollama is installed
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        print("✅ Ollama is installed.")
    except subprocess.CalledProcessError:
        raise RuntimeError("❌ Ollama is not installed or not in PATH.")

    # Download the model
    print("⬇️ Downloading Qwen3 model via Ollama...")
    subprocess.run(["ollama", "pull", model], check=True)
    print("✅ Qwen3 model downloaded and ready to use with Ollama.")

if __name__ == "__main__":
    download_qwen_model()
