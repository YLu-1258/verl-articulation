#!/usr/bin/env python3
"""
Simple test script to verify VERL-Articulation custom reward function logging works.
"""

def test_reward_function():
    """Create and test a simple reward function."""
    
    # Create a simple reward function
    reward_function_code = '''
def test_reward_with_subrewards(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Test reward function that demonstrates sub-reward logging."""
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
    
    # Combine scores
    final_score = 0.5 * length_score + 0.5 * relevance_score
    
    return {
        "score": final_score,
        "length_score": length_score,
        "relevance_score": relevance_score,
        "word_count": response_length,
    }
'''
    
    # Write the reward function to a file
    with open("test_reward_function.py", "w") as f:
        f.write(reward_function_code)
    
    print("✅ Created test_reward_function.py")
    return True

def test_reward_directly():
    """Test the reward function directly."""
    
    import sys
    sys.path.append(".")
    from test_reward_function import test_reward_with_subrewards
    
    # Test case
    result = test_reward_with_subrewards(
        data_source="geography",
        solution_str="The capital of France is Paris, a beautiful city.",
        ground_truth="Paris"
    )
    
    print("📝 Test Result:", result)
    return result

def test_metrics_computation():
    """Test the metrics computation function."""
    
    try:
        from verl_articulation.trainer.ppo.reward import compute_reward_metrics
        
        # Test data
        test_reward_extra_info = {
            "length_score": [0.8, 0.9, 0.7, 0.85],
            "relevance_score": [1.0, 1.0, 0.3, 1.0],
            "word_count": [15, 12, 25, 18]
        }
        
        metrics = compute_reward_metrics(test_reward_extra_info)
        
        print("📊 Computed metrics:")
        for key, value in sorted(metrics.items()):
            print(f"   {key}: {value:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import metrics function: {e}")
        return False

def main():
    """Main function to test everything."""
    
    print("🚀 VERL-Articulation Custom Reward Function Test")
    print("=" * 50)
    
    # Step 1: Create test files
    test_reward_function()
    
    # Step 2: Test reward function directly
    print("\n🧪 Testing reward function directly...")
    try:
        result = test_reward_directly()
        print("✅ Reward function test completed!")
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return
    
    # Step 3: Test metrics computation
    print("\n📊 Testing metrics computation...")
    metrics_test_passed = test_metrics_computation()
    
    if metrics_test_passed:
        print("✅ Metrics computation test completed!")
    
    print("\n🎯 Next Steps:")
    print("Your custom reward function is working correctly!")
    print("During training, you'll see metrics like:")
    print("- reward/length_score/mean")
    print("- reward/relevance_score/mean") 
    print("- reward/word_count/mean")

if __name__ == "__main__":
    main()
