# Custom Reward Function with Sub-Reward Logging

## Overview

This enhancement allows custom reward functions to return multiple sub-rewards that are automatically logged as separate metrics during training and validation. This is useful when you want to track individual components of your reward function (e.g., fluency, relevance, safety scores) rather than just the final combined reward.

## Changes Made

### 1. New Function in `verl/trainer/ppo/reward.py`

Added `compute_reward_metrics()` function that:
- Takes the `reward_extra_info` dictionary returned by reward managers  
- Computes statistics (mean, max, min, std) for each sub-reward
- Returns a dictionary of metrics with keys like `reward/{sub_reward_name}/mean`

### 2. Updated Training Loop in `verl/trainer/ppo/ray_trainer.py`

Modified the training loop to:
- Import the new `compute_reward_metrics` function
- Call it when `reward_extra_infos_dict` is available
- Add the computed metrics to the main metrics dictionary for logging
- Handle both async and non-async reward computation

### 3. Updated Validation Loop

Enhanced validation to also compute and log sub-reward metrics with "val-" prefix.

## How to Use

### 1. Create Your Custom Reward Function

Your reward function should follow VERL's standard API and return a dictionary with sub-rewards:

```python
def my_custom_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Custom reward function following VERL's API.
    
    Args:
        data_source: Dataset identifier (e.g., "openai/gsm8k") 
        solution_str: The model's generated response string
        ground_truth: Expected answer string
        extra_info: Optional dict with additional info (num_turns, etc.)
        **kwargs: Additional parameters from reward_kwargs in config
    
    Returns:
        dict: Dictionary with 'score' key and optional sub-reward keys
    """
    # Your reward computation logic here
    fluency_score = compute_fluency(solution_str)
    relevance_score = compute_relevance(solution_str, ground_truth) 
    safety_score = compute_safety(solution_str)
    
    final_score = combine_rewards(fluency_score, relevance_score, safety_score)
    
    # Return dictionary for sub-reward logging
    return {
        "score": final_score,        # Main reward used for training (required)
        "fluency": fluency_score,    # Sub-rewards that will be logged
        "relevance": relevance_score,
        "safety": safety_score,
    }
```

**Important**: The function signature must match VERL's API:
- `data_source` (str): Dataset identifier  
- `solution_str` (str): Model's response
- `ground_truth` (str): Expected answer
- `extra_info` (dict, optional): Additional metadata

### 2. Configure Your Training

Add the custom reward function to your config:

```yaml
custom_reward_function:
  path: "/path/to/your/reward_function.py"
  name: "my_custom_reward"  # Optional if function is named 'compute_score'
  reward_kwargs:
    # Any additional arguments for your function
    weight_fluency: 0.3
    weight_relevance: 0.5
```

### 3. Monitor Your Metrics

During training, you'll see these metrics logged:
- `reward/fluency/mean`, `reward/fluency/max`, `reward/fluency/min`, `reward/fluency/std`
- `reward/relevance/mean`, `reward/relevance/max`, `reward/relevance/min`, `reward/relevance/std`
- `reward/safety/mean`, `reward/safety/max`, `reward/safety/min`, `reward/safety/std`
- `reward/score/mean`, `reward/score/max`, `reward/score/min`, `reward/score/std`

During validation:
- `val-reward/fluency/mean`, `val-reward/fluency/max`, etc.

## Backward Compatibility

If your reward function returns just a float/int (the old way), it will continue to work without any changes. The sub-reward logging is only enabled when your function returns a dictionary with additional keys beyond just the main score.

## Data Types Supported

The metrics computation handles various data types:
- Numeric values (int, float)
- Lists and tuples (flattened and converted to numpy arrays)  
- Nested structures (automatically flattened)

Non-numeric data is automatically skipped.

## Key Differences from Previous Assumptions

**What we initially assumed:**
- Custom reward function receives `DataProto` object
- Function returns dictionary with `reward_tensor` and `reward_extra_info`

**What VERL actually expects:**
- Custom reward function receives string parameters (`data_source`, `solution_str`, `ground_truth`, `extra_info`)
- Function returns either float/int OR dictionary with `score` key + sub-reward keys
- The reward manager handles the `DataProto` processing and calls your function per sample

## Examples

See:
- `examples/custom_reward_function_example.py` - Example reward function implementations
- `examples/custom_reward_config_example.yaml` - Configuration examples

## Integration Point

The custom reward function integrates into VERL through:
1. `load_reward_manager()` in `reward.py` loads your custom function
2. Reward manager (e.g., `NaiveRewardManager`) calls your function for each sample
3. Manager collects results and returns `reward_extra_info` dictionary
4. Training loop calls `compute_reward_metrics()` on this dictionary
5. Metrics are logged to your configured logger (wandb, tensorboard, etc.)
