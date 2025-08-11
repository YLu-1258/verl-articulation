#!/usr/bin/env python3
"""
Simple test script to validate the reward metrics computation logic.
This can be run independently to test the function.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import torch
    
    def compute_reward_metrics(reward_extra_infos_dict):
        """Test version of the compute_reward_metrics function."""
        metrics = {}
        
        for key, values in reward_extra_infos_dict.items():
            if not values:  # Skip empty lists
                continue
                
            try:
                # Convert to numpy array for easier computation
                if isinstance(values[0], torch.Tensor):
                    values_array = torch.cat([v.flatten() for v in values])
                    values_array = values_array.detach().cpu().numpy()
                elif isinstance(values[0], (list, tuple)):
                    values_array = np.array([item for sublist in values for item in sublist])
                else:
                    values_array = np.array(values)
                
                # Skip non-numeric values or empty arrays
                if not np.issubdtype(values_array.dtype, np.number) or len(values_array) == 0:
                    continue
                    
                # Handle NaN and infinite values
                values_array = values_array[np.isfinite(values_array)]
                if len(values_array) == 0:
                    continue
                    
                # Compute statistics
                metrics[f"reward/{key}/mean"] = float(np.mean(values_array))
                metrics[f"reward/{key}/max"] = float(np.max(values_array))
                metrics[f"reward/{key}/min"] = float(np.min(values_array))
                
                if len(values_array) > 1:
                    metrics[f"reward/{key}/std"] = float(np.std(values_array))
                    
            except Exception as e:
                # Log warning but don't fail the training
                print(f"Warning: Could not compute metrics for reward key '{key}': {e}")
                continue
        
        return metrics

    def test_reward_metrics():
        """Test the reward metrics computation with various data types."""
        
        # Test case 1: Simple lists
        test_data_1 = {
            "fluency": [0.8, 0.9, 0.7, 0.85],
            "relevance": [0.6, 0.8, 0.9, 0.7],
            "safety": [0.9, 0.95, 0.88, 0.92]
        }
        
        metrics_1 = compute_reward_metrics(test_data_1)
        print("Test 1 - Simple lists:")
        for key, value in sorted(metrics_1.items()):
            print(f"  {key}: {value:.4f}")
        print()
        
        # Test case 2: Torch tensors
        test_data_2 = {
            "tensor_reward": [
                torch.tensor([0.5, 0.6]), 
                torch.tensor([0.8, 0.9]),
                torch.tensor([0.7, 0.75])
            ]
        }
        
        metrics_2 = compute_reward_metrics(test_data_2)
        print("Test 2 - Torch tensors:")
        for key, value in sorted(metrics_2.items()):
            print(f"  {key}: {value:.4f}")
        print()
        
        # Test case 3: Nested lists
        test_data_3 = {
            "nested_scores": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }
        
        metrics_3 = compute_reward_metrics(test_data_3)
        print("Test 3 - Nested lists:")
        for key, value in sorted(metrics_3.items()):
            print(f"  {key}: {value:.4f}")
        print()
        
        # Test case 4: Edge cases
        test_data_4 = {
            "empty_list": [],
            "string_data": ["hello", "world"],
            "with_nan": [0.5, float('nan'), 0.7],
            "with_inf": [0.5, float('inf'), 0.7],
            "single_value": [0.42]
        }
        
        metrics_4 = compute_reward_metrics(test_data_4)
        print("Test 4 - Edge cases:")
        for key, value in sorted(metrics_4.items()):
            print(f"  {key}: {value:.4f}")
        print()
        
        print("All tests completed!")

    if __name__ == "__main__":
        test_reward_metrics()
        
except ImportError as e:
    print(f"Required packages not available: {e}")
    print("This test requires numpy and torch to run.")
