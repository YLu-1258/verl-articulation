#!/usr/bin/env python3
"""
Quick test script to verify VERL-Articulation custom reward function logging works.
This creates a minimal example you can run to test the functionality.
"""

def create_test_reward_function():
    """Create a test reward function file."""
    
    reward_function_code = '''#!/usr/bin/env python3

def test_reward_with_subrewards(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Test reward function that demonstrates sub-reward logging.
    """
    # Simple sub-reward calculations
    response_length = len(solution_str.split())
    
    # Length score (prefer 10-50 words)
    if response_length < 10:
        length_score = response_length / 10.0
    elif response_length > 50:
        length_score = 50.0 / response_length
    else:
        length_score = 1.0
    
    # Relevance score (check if ground truth keywords are in response)
    if ground_truth and any(word.lower() in solution_str.lower() for word in ground_truth.split()):
        relevance_score = 1.0
    else:
        relevance_score = 0.3
    
    # Safety score (avoid negative words)
    negative_words = ["bad", "terrible", "horrible", "awful"]
    safety_score = 0.2 if any(word in solution_str.lower() for word in negative_words) else 1.0
    
    # Combine scores
    final_score = 0.3 * length_score + 0.5 * relevance_score + 0.2 * safety_score
    
    return {
        "score": final_score,
        "length_score": length_score,
        "relevance_score": relevance_score,
        "safety_score": safety_score,
        "word_count": response_length,
    }

# Default function name (optional)
compute_score = test_reward_with_subrewards
'''
    
    with open("test_reward_function.py", "w") as f:
        f.write(reward_function_code)
    
    print("✅ Created test_reward_function.py")

def create_test_dataset():
    """Create a simple test dataset."""
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not available. Install with: pip install pandas")
        return False
    
    # Create simple test data
    data = [
        {
            "prompt": "What is the capital of France?",
            "ground_truth": "Paris",
            "data_source": "geography"
        },
        {
            "prompt": "What is 5 + 3?",
            "ground_truth": "8",
            "data_source": "math"
        },
        {
            "prompt": "Name a primary color.",
            "ground_truth": "red blue yellow",
            "data_source": "colors"
        },
        {
            "prompt": "What animal says 'moo'?",
            "ground_truth": "cow",
            "data_source": "animals"
        }
    ]
    
    df = pd.DataFrame(data)
    df.to_parquet("test_dataset.parquet", index=False)
    print("✅ Created test_dataset.parquet")
    return True

def create_minimal_config():
    """Create a minimal configuration file."""
    
    config_content = '''# Minimal VERL-Articulation Configuration for Testing Custom Reward Function

# Model configuration (using a very small model for testing)
actor_rollout_ref:
  model:
    path: "gpt2"  # Small model for quick testing
    lora_rank: 0
  actor:
    strategy: "fsdp"
    micro_batch_size_per_gpu: 1
    grad_accum_steps: 1
  rollout:
    n: 2  # Generate 2 responses per prompt
    mode: "sync"

# Disable critic for simpler testing
critic:
  enable: false

# Custom reward function configuration
custom_reward_function:
  path: "./test_reward_function.py"
  name: "test_reward_with_subrewards"
  reward_kwargs: {}

# Reward model configuration
reward_model:
  enable: false
  launch_reward_fn_async: false

# Data configuration
data:
  train_files: ["test_dataset.parquet"]
  val_files: ["test_dataset.parquet"]
  train_batch_size: 2
  val_batch_size: 2
  shuffle: false
  reward_fn_key: "data_source"

# Algorithm configuration
algorithm:
  adv_estimator: "REINFORCE"  # Simplest advantage estimator
  use_kl_in_reward: false

# Training configuration
trainer:
  total_epochs: 1  # Just one epoch for testing
  n_gpus_per_node: 1
  nnodes: 1
  project_name: "verl_articulation_test"
  experiment_name: "custom_reward_test"
  logger: ["console"]
  log_val_generations: 2

# Ray configuration
ray_init:
  num_cpus: 2
'''
    
    with open("test_config.yaml", "w") as f:
        f.write(config_content)
    
    print("✅ Created test_config.yaml")

def test_reward_function_directly():
    """Test the reward function directly without running full training."""
    
    print("\n🧪 Testing reward function directly...")
    
    # Import the reward function
    try:
        import sys
        sys.path.append(".")
        from test_reward_function import test_reward_with_subrewards
        
        # Test cases
        test_cases = [
            {
                "data_source": "geography",
                "solution_str": "The capital of France is Paris, a beautiful city.",
                "ground_truth": "Paris"
            },
            {
                "data_source": "math", 
                "solution_str": "5 + 3 equals 8",
                "ground_truth": "8"
            },
            {
                "data_source": "colors",
                "solution_str": "That's a terrible question. I don't know.",
                "ground_truth": "red blue yellow"
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test Case {i}:")
            print(f"   Question type: {case['data_source']}")
            print(f"   Response: {case['solution_str']}")
            print(f"   Ground truth: {case['ground_truth']}")
            
            result = test_reward_with_subrewards(**case)
            print(f"   Results: {result}")
        
        print("\n✅ Reward function test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Could not import reward function: {e}")
        return False

def test_metrics_computation():
    """Test the metrics computation function."""
    
    print("\n📊 Testing metrics computation...")
    
    try:
        # Try to import the metrics function
        import sys
        import os
        
        # Add the verl_articulation path
        verl_path = None
        for root, dirs, files in os.walk("."):
            if "verl_articulation" in dirs and os.path.exists(os.path.join(root, "verl_articulation", "trainer")):
                verl_path = root
                break
        
        if verl_path:
            sys.path.insert(0, verl_path)
            from verl_articulation.trainer.ppo.reward import compute_reward_metrics
            
            # Test data
            test_reward_extra_info = {
                "length_score": [0.8, 0.9, 0.7, 0.85],
                "relevance_score": [1.0, 1.0, 0.3, 1.0],
                "safety_score": [1.0, 1.0, 0.2, 1.0],
                "word_count": [15, 12, 25, 18]
            }
            
            metrics = compute_reward_metrics(test_reward_extra_info)
            
            print("   Computed metrics:")
            for key, value in sorted(metrics.items()):
                print(f"     {key}: {value:.4f}")
            
            print("\n✅ Metrics computation test completed!")
            return True
            
        else:
            print("❌ Could not find VERL-Articulation installation")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import metrics function: {e}")
        print("   This is expected if VERL-Articulation is not installed yet.")
        return False

def main():
    """Main function to set up and test everything."""
    
    print("🚀 VERL-Articulation Custom Reward Function Test Setup")
    print("=" * 60)
    
    # Step 1: Create test files
    create_test_reward_function()
    
    dataset_created = create_test_dataset()
    if not dataset_created:
        return
    
    create_minimal_config()
    
    # Step 2: Test reward function directly
    reward_test_passed = test_reward_function_directly()
    
    # Step 3: Test metrics computation
    metrics_test_passed = test_metrics_computation()
    
    # Step 4: Instructions for running full training
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Install VERL-Articulation: pip install -e .")
    print("2. Install dependencies: pip install torch transformers datasets ray omegaconf hydra-core pandas")
    print("3. Run training: python -m verl_articulation.trainer.main_ppo --config-path=. --config-name=test_config")
    
    if reward_test_passed:
        print("\n✅ Your reward function is working correctly!")
        print("   During training, you'll see metrics like:")
        print("   - reward/length_score/mean")
        print("   - reward/relevance_score/mean") 
        print("   - reward/safety_score/mean")
        print("   - reward/word_count/mean")
    
    if metrics_test_passed:
        print("\n✅ Metrics computation is working correctly!")
    else:
        print("\n⚠️  Install VERL-Articulation to test metrics computation.")
    
    print("\nFiles created:")
    print("- test_reward_function.py (your custom reward function)")
    print("- test_dataset.parquet (sample training data)")
    print("- test_config.yaml (VERL-Articulation configuration)")

if __name__ == "__main__":
    main()
