# model_selection_cli
# Hyperbolic Model Comparison CLI

A command-line tool to compare different LLM models hosted on the Hyperbolic API.

## Setup

1. Clone this repository
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your Hyperbolic API key:
   ```
   HYPERBOLIC_API_KEY=your_api_key_here
   ```

## Usage

Compare two models using the CLI:

```bash
python .venv/cli/optionb.py model1_id model2_id [--prompt "Your test prompt"]
```

Example:
```bash
python .venv/cli/optionb.py deepseek-ai/DeepSeek-V3-0324 Qwen/QwQ-32B
```

### Options

- `--prompt`: Specify a custom prompt for testing (default: "Explain quantum computing in simple terms.")

### Output

The tool will display a comparison table with the following metrics:

- Speed Metrics:
  - Time to first token (ms)
  - Total latency (ms)
  - Tokens per second

- Cost Analysis:
  - Input cost ($/1K tokens)
  - Output cost ($/1K tokens)
  - Cost-performance ratio

## Notes

- The tool sets `temperature` and `top_p` to 0.0 for consistent comparison
- All API calls are made asynchronously for better performance
- The cost-performance ratio is calculated based on speed and cost metrics 
