# 🚀 VERL Custom Reward Function Usage Guide

This guide shows you how to use VERL (the reinforcement learning library) with custom reward functions that log sub-reward metrics.

## 📋 Prerequisites

1. **Install VERL**:
   ```bash
   pip install -e .  # From the verl-articulation directory
   # OR
   pip install verl  # If installing from PyPI
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers datasets ray omegaconf hydra-core
   ```

## 🎯 Step-by-Step Usage

### Step 1: Create Your Custom Reward Function

Create a file `my_reward_function.py`:

```python
#!/usr/bin/env python3

def my_custom_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Custom reward function following VERL's API.
    
    Args:
        data_source (str): Dataset identifier (e.g., "openai/gsm8k")
        solution_str (str): Model's generated response
        ground_truth (str): Expected answer
        extra_info (dict): Additional metadata (may include num_turns, etc.)
        **kwargs: Extra parameters from reward_kwargs in config
        
    Returns:
        dict: Dictionary with 'score' key + sub-reward keys for logging
    """
    import re
    
    # Example sub-reward 1: Length appropriateness
    response_words = len(solution_str.split())
    min_words = kwargs.get('min_words', 5)
    max_words = kwargs.get('max_words', 100)
    
    if response_words < min_words:
        length_score = response_words / min_words
    elif response_words > max_words:
        length_score = max_words / response_words
    else:
        length_score = 1.0
    
    # Example sub-reward 2: Relevance (simple keyword matching)
    relevance_score = 0.5  # Default
    if ground_truth and ground_truth.lower() in solution_str.lower():
        relevance_score = 1.0
    elif any(word in solution_str.lower() for word in ground_truth.lower().split()[:3]):
        relevance_score = 0.7
    
    # Example sub-reward 3: Format compliance
    # Check if response has proper structure (this is dataset-specific)
    format_score = 1.0
    if data_source == "openai/gsm8k":
        # GSM8K expects answer after "####"
        format_score = 1.0 if "####" in solution_str else 0.1
    
    # Example sub-reward 4: Safety (placeholder)
    safety_words = ["harmful", "dangerous", "illegal", "violence"]
    safety_score = 0.0 if any(word in solution_str.lower() for word in safety_words) else 1.0
    
    # Combine sub-rewards into final score
    weights = kwargs.get('weights', {'length': 0.2, 'relevance': 0.5, 'format': 0.2, 'safety': 0.1})
    
    final_score = (
        weights['length'] * length_score +
        weights['relevance'] * relevance_score +
        weights['format'] * format_score +
        weights['safety'] * safety_score
    )
    
    # Return dictionary - 'score' is required, others will be logged automatically
    return {
        "score": final_score,              # Main reward used for training
        "length_score": length_score,      # Logged as reward/length_score/*
        "relevance_score": relevance_score, # Logged as reward/relevance_score/*
        "format_score": format_score,      # Logged as reward/format_score/*
        "safety_score": safety_score,      # Logged as reward/safety_score/*
        "response_length": response_words, # Logged as reward/response_length/*
    }

# Alternative: Simple reward function (backward compatible)
def simple_reward(data_source, solution_str, ground_truth, extra_info=None):
    """Simple reward function that returns just a score."""
    # Simple length-based reward
    return min(len(solution_str.split()) / 20.0, 1.0)
```

### Step 2: Prepare Your Dataset

VERL expects data in specific formats. Here's an example for GSM8K:

```python
# create_dataset.py
import pandas as pd

# Example dataset creation
data = [
    {
        "prompt": "What is 2 + 2?",
        "ground_truth": "4",
        "data_source": "simple_math"
    },
    {
        "prompt": "If I have 5 apples and eat 2, how many are left?", 
        "ground_truth": "3",
        "data_source": "simple_math"
    }
]

df = pd.DataFrame(data)
df.to_parquet("my_dataset.parquet", index=False)
print("Dataset created: my_dataset.parquet")
```

### Step 3: Create Configuration File

Create `my_ppo_config.yaml`:

```yaml
# Basic PPO Configuration with Custom Reward Function

# Model and training setup
actor_rollout_ref:
  model:
    path: "microsoft/DialoGPT-small"  # Example small model for testing
    lora_rank: 0  # Set to > 0 to use LoRA
  actor:
    strategy: "fsdp"
    micro_batch_size_per_gpu: 2
    grad_accum_steps: 1
  rollout:
    n: 4  # Number of responses per prompt
    mode: "sync"

# Critic setup (optional - can disable for simpler testing)
critic:
  enable: true
  strategy: "fsdp" 
  micro_batch_size_per_gpu: 2

# Custom reward function configuration
custom_reward_function:
  path: "./my_reward_function.py"          # Path to your reward function
  name: "my_custom_reward"                 # Function name in the file
  reward_kwargs:                           # Arguments passed to your function
    min_words: 5
    max_words: 100
    weights:
      length: 0.2
      relevance: 0.5
      format: 0.2
      safety: 0.1

# Reward model configuration
reward_model:
  enable: false                            # Using custom function instead
  launch_reward_fn_async: false           # Can be true for better performance

# Data configuration
data:
  train_files: ["my_dataset.parquet"]     # Your training data
  val_files: ["my_dataset.parquet"]       # Your validation data
  train_batch_size: 4
  val_batch_size: 2
  shuffle: true
  reward_fn_key: "data_source"            # Column name for dataset identifier

# Algorithm configuration
algorithm:
  adv_estimator: "GAE"                    # Generalized Advantage Estimation
  use_kl_in_reward: false                 # Enable KL penalty if needed

# Training configuration
trainer:
  total_epochs: 2                         # Number of training epochs
  n_gpus_per_node: 1                      # Number of GPUs per node
  nnodes: 1                               # Number of nodes
  project_name: "my_verl_experiment"      # For logging
  experiment_name: "custom_reward_test"   # For logging
  logger: ["console"]                     # Can add "wandb", "tensorboard"
  log_val_generations: 5                  # Number of validation samples to log

# Ray configuration
ray_init:
  num_cpus: 4                             # CPU cores for Ray
```

### Step 4: Run Training

```bash
# Make sure you're in the verl directory
cd /data/alexl/verl-articulation

# Run PPO training with your configuration
python -m verl_articulation.trainer.main_ppo --config-path=. --config-name=my_ppo_config

# Alternative: Use hydra directly
python verl/trainer/main_ppo.py --config-path=. --config-name=my_ppo_config
```

## 📊 What You'll See

With the custom reward logging implemented, you'll see these metrics during training:

### Training Metrics:
- `reward/length_score/mean` - Average length score across batch
- `reward/length_score/max` - Maximum length score  
- `reward/length_score/min` - Minimum length score
- `reward/length_score/std` - Standard deviation of length scores
- `reward/relevance_score/mean` - Average relevance score
- `reward/format_score/mean` - Average format compliance score
- `reward/safety_score/mean` - Average safety score
- `reward/response_length/mean` - Average response length

### Validation Metrics:
- `val-reward/length_score/mean` - Same metrics with val- prefix
- `val-reward/relevance_score/mean`
- etc.

## 🎛️ Advanced Configuration

### Adding W&B Logging:
```yaml
trainer:
  logger: ["console", "wandb"]
  
# Add wandb config
wandb:
  project: "my-verl-project"
  entity: "my-team"
```

### Using Multiple Datasets:
```yaml
data:
  train_files: 
    - "dataset1.parquet"
    - "dataset2.parquet"
  # Different reward functions can be used based on data_source column
```

### Enabling KL Penalty:
```yaml
algorithm:
  use_kl_in_reward: true
  kl_penalty: "kl"  # or "abs", "mse" 
```

## 🔧 Troubleshooting

1. **Import Errors**: Make sure VERL is properly installed with `pip install -e .`

2. **GPU Memory Issues**: Reduce `micro_batch_size_per_gpu` or use gradient checkpointing

3. **Ray Issues**: Check Ray cluster status with `ray status`

4. **Custom Function Not Found**: Verify the path and function name in your config

5. **No Sub-Reward Metrics**: Ensure your reward function returns a dictionary with 'score' key

## 🎯 Quick Test

Run this minimal test to verify everything works:

```bash
# Create a simple test
python -c "
from verl_articulation.trainer.ppo.reward import compute_reward_metrics

# Test metrics computation
test_data = {
    'length_score': [0.8, 0.9, 0.7],
    'relevance_score': [0.6, 0.8, 0.9]
}

metrics = compute_reward_metrics(test_data)
print('Computed metrics:', metrics)
"
```

## 📚 Next Steps

1. **Scale Up**: Use larger models and datasets
2. **Experiment**: Try different reward function combinations
3. **Monitor**: Use W&B or TensorBoard for detailed tracking
4. **Optimize**: Tune hyperparameters based on sub-reward metrics

The custom reward logging feature will help you understand how each component of your reward function performs during training!
