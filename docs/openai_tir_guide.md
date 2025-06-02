# Using OpenAI Models with TIR (Tool-Integrated Reasoning)

This guide explains how to use OpenAI models with the TIR (Tool-Integrated Reasoning) pipeline in NeMo-Skills after the recent modifications that enable code execution support for OpenAI servers.

## Overview

Previously, the TIR pipeline only supported local models (TRTLLM, vLLM, etc.) because the code execution wrapper couldn't handle OpenAI's chat message format. We've now modified the `CodeExecutionWrapper` to support both string prompts (for local models) and dictionary prompts (for OpenAI models).

## Key Changes Made

1. **Modified `CodeExecutionWrapper`**: Updated `_generate_single` and `_stream_single` methods to handle OpenAI-style dictionary prompts
2. **Removed OpenAI Block**: Removed the hard-coded check that prevented OpenAI servers from using code execution
3. **Format Detection**: Added automatic detection of prompt format (string vs. dictionary) to handle both local and OpenAI models seamlessly

## Configuration

### Basic OpenAI TIR Configuration

```yaml
# Use the existing tir-openmath.yaml as a base, with these key settings:
stages:
  generate_solutions:
    inline_args: >-
      ++prompt_config=openmath/tir
      ++inference.tokens_to_generate=16384
      ++code_execution=true
      ++server.code_execution.max_code_executions=null
      ++server.code_execution.add_remaining_code_executions=true
      ++total_code_executions_in_prompt='[1, 8]'
      ++override_max_code_executions=true
    stage_kwargs:
      model: gpt-4.1-mini  # or gpt-4o, gpt-4o-mini, etc.
      server_type: openai
      server_address: https://api.openai.com/v1
      num_random_seeds: 16
      with_sandbox: true  # IMPORTANT: Enable sandbox for code execution
```

### Environment Setup

1. **Set API Key**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. **Activate NeMo Skills Environment**:
   ```bash
   source ~/nemo-skills-env/bin/activate
   ```

## Running TIR with OpenAI Models

### Option 1: Use the Modified Configuration

```bash
python recipes/openmathreasoning/pipeline/solution_generation.py --mode tir-openmath
```

### Option 2: Custom Configuration

Create your own configuration file based on `tir-openmath.yaml`:

```yaml
# custom-openai-tir.yaml
cluster: local
base_output_dir: /workspace/openai-tir-demo
expname: openai-tir
suffix: openai

# ... (copy other settings from tir-openmath.yaml)

stages:
  generate_solutions:
    # ... other settings ...
    stage_kwargs:
      model: gpt-4o-mini  # Choose your preferred OpenAI model
      server_type: openai
      server_address: https://api.openai.com/v1
      with_sandbox: true
```

Then run:
```bash
python recipes/openmathreasoning/pipeline/solution_generation.py --config-path=path/to/custom-openai-tir.yaml
```

## Supported OpenAI Models

The following OpenAI models are supported:
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.1-mini` (reasoning model with special parameter filtering)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

## Code Execution Details

### How It Works

1. **Prompt Format**: OpenAI models use chat message format with `[{"role": "user", "content": "..."}]`
2. **Code Detection**: The system looks for code blocks between `<tool_call>` and `</tool_call>` tags
3. **Execution**: Python code is executed in a local sandbox
4. **Result Integration**: Code output is added to the conversation using `<tool_output>` tags
5. **Continuation**: The model continues generating based on the code results

### Example Flow

```
User: Calculate the sum of squares from 1 to 10.

Model: I'll calculate the sum of squares from 1 to 10 using Python.

<tool_call>
sum_of_squares = sum(i**2 for i in range(1, 11))
print(f"Sum of squares from 1 to 10: {sum_of_squares}")
</tool_call>

<tool_output>
Sum of squares from 1 to 10: 385
</tool_output>

The sum of squares from 1 to 10 is 385.
```

## Troubleshooting

### Common Issues

1. **"Code execution is not supported for OpenAI server"**
   - This error indicates you're using an older version. Make sure you have the updated code.

2. **"Invalid OpenAI prompt format"**
   - Ensure your prompts are in the correct chat message format with `role` and `content` fields.

3. **Sandbox not starting**
   - Make sure `with_sandbox: true` is set in your configuration
   - Check that you have the necessary permissions to run local sandbox

4. **API Rate Limits**
   - OpenAI has rate limits. Consider reducing `num_random_seeds` or adding delays
   - Use `gpt-4o-mini` for cost-effective testing

### Performance Considerations

- **Cost**: OpenAI models charge per token. TIR can generate long sequences with code execution
- **Speed**: API calls are slower than local models but don't require GPU resources
- **Rate Limits**: Be mindful of OpenAI's rate limits when running large batches

## Advanced Configuration

### Custom Code Execution Settings

```yaml
stages:
  generate_solutions:
    inline_args: >-
      ++code_execution=true
      ++server.code_execution.max_code_executions=5
      ++server.code_execution.code_execution_timeout=15.0
      ++server.code_execution.max_code_output_characters=2000
      ++server.code_execution.add_remaining_code_executions=true
```

### Using Different Code Block Tags

```yaml
postprocess_tir_generations:
  code_begin: "```python\n"
  code_end: "```\n"
```

## Testing Your Setup

Use the provided test script to verify your setup:

```bash
python test_openai_code_execution.py
```

This will test basic code execution functionality with OpenAI models.

## Limitations

1. **Streaming**: Streaming support is available but may have different behavior compared to local models
2. **Model-specific features**: Some OpenAI-specific features may not be fully supported
3. **Cost**: Be aware of API costs when running large-scale experiments

## Migration from Local Models

If you're migrating from local TIR models to OpenAI:

1. Update your configuration to use `server_type: openai`
2. Set your OpenAI API key
3. Enable sandbox with `with_sandbox: true`
4. Adjust batch sizes and rate limits as needed
5. Test with a small dataset first

## Support

For issues specific to OpenAI TIR integration, check:
1. Your API key is correctly set
2. The sandbox is properly configured
3. Your prompt format is correct
4. Rate limits are not exceeded

For general NeMo-Skills issues, refer to the main documentation. 