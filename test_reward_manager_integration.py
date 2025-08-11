#!/usr/bin/env python3
"""
Test script to verify that the reward manager correctly processes custom reward functions
and returns the expected extra info for logging.
"""

def test_reward_manager_integration():
    """Test that demonstrates how the reward manager processes custom reward functions."""
    
    # Mock custom reward function that returns dictionary with sub-rewards
    def mock_custom_reward(data_source, solution_str, ground_truth, extra_info=None):
        """Mock reward function following VERL's API."""
        response_length = len(solution_str.split())
        
        # Compute some mock sub-rewards
        fluency_score = min(response_length / 20, 1.0)
        relevance_score = 0.8 if ground_truth in solution_str else 0.3
        safety_score = 0.9  # Assume mostly safe
        
        # Combine into final score
        final_score = 0.3 * fluency_score + 0.5 * relevance_score + 0.2 * safety_score
        
        # Return dictionary with sub-rewards
        return {
            "score": final_score,
            "fluency": fluency_score,
            "relevance": relevance_score,  
            "safety": safety_score,
            "response_length": response_length,
        }
    
    # Mock tokenizer (simplified)
    class MockTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            # For testing, just return a mock string based on length
            return " ".join([f"token_{i}" for i in range(len(token_ids))])
    
    # Test the reward manager behavior
    print("Testing reward manager integration...")
    print("="*60)
    
    # Simulate how the reward manager would process the custom function
    tokenizer = MockTokenizer()
    
    # Mock data similar to what the reward manager receives
    test_cases = [
        {
            "data_source": "test_dataset",
            "response_tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ground_truth": "token_5 token_6",
        },
        {
            "data_source": "test_dataset", 
            "response_tokens": [1, 2, 3],
            "ground_truth": "different answer",
        }
    ]
    
    reward_extra_info = {}
    
    for i, test_case in enumerate(test_cases):
        # Simulate what the reward manager does
        response_str = tokenizer.decode(test_case["response_tokens"])
        
        print(f"Sample {i+1}:")
        print(f"  Response: {response_str}")
        print(f"  Ground Truth: {test_case['ground_truth']}")
        
        # Call custom reward function
        result = mock_custom_reward(
            data_source=test_case["data_source"],
            solution_str=response_str,
            ground_truth=test_case["ground_truth"],
        )
        
        print(f"  Reward Result: {result}")
        
        # Collect extra info (this is what the reward manager does)
        for key, value in result.items():
            if key not in reward_extra_info:
                reward_extra_info[key] = []
            reward_extra_info[key].append(value)
        
        print()
    
    print("Collected reward_extra_info:")
    for key, values in reward_extra_info.items():
        print(f"  {key}: {values}")
    
    print()
    print("This is what would be passed to compute_reward_metrics():")
    
    # Simulate what our compute_reward_metrics function would do
    metrics = {}
    
    for key, values in reward_extra_info.items():
        if key == "score":
            continue  # Skip the main score
        
        import statistics
        try:
            metrics[f"reward/{key}/mean"] = statistics.mean(values)
            metrics[f"reward/{key}/max"] = max(values)
            metrics[f"reward/{key}/min"] = min(values)
            if len(values) > 1:
                metrics[f"reward/{key}/std"] = statistics.stdev(values)
        except:
            print(f"  Skipping non-numeric key: {key}")
    
    print("\nComputed metrics:")
    for metric_name, metric_value in sorted(metrics.items()):
        print(f"  {metric_name}: {metric_value:.4f}")
    
    print("\n" + "="*60)
    print("✅ Test completed! This shows how custom reward functions")
    print("   with dictionary returns get processed into logged metrics.")


if __name__ == "__main__":
    test_reward_manager_integration()
