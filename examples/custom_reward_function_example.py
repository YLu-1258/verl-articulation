#!/usr/bin/env python3
"""
Example custom reward function that demonstrates how to return multiple sub-rewards
that will be automatically logged by the VERL trainer.

This example shows a reward function that computes:
1. A fluency score based on response length
2. A relevance score based on keyword matching  
3. A safety score (placeholder)
4. A combined final reward as a weighted sum

All of these metrics will be logged separately with statistics (mean, max, min, std).

IMPORTANT: The custom reward function should follow VERL's API:
- Parameters: data_source, solution_str, ground_truth, extra_info=None
- Return: Either a float/int OR a dictionary with 'score' key + extra keys for sub-rewards
"""

def custom_reward_with_subrewards(
    data_source: str,
    solution_str: str, 
    ground_truth: str,
    extra_info=None,
    # Custom parameters can be passed via reward_kwargs in config
    fluency_weight: float = 0.3,
    relevance_weight: float = 0.5,
    safety_weight: float = 0.2,
    min_length: int = 10,
    max_length: int = 200,
):
    """
    Custom reward function that computes multiple sub-rewards.
    
    This follows VERL's reward function API with data_source, solution_str, ground_truth parameters.
    
    Args:
        data_source: The dataset name/identifier
        solution_str: The model's response/solution string  
        ground_truth: The expected ground truth answer
        extra_info: Additional information (may contain num_turns, etc.)
        fluency_weight: Weight for fluency component
        relevance_weight: Weight for relevance component  
        safety_weight: Weight for safety component
        min_length: Minimum desired response length
        max_length: Maximum desired response length
        
    Returns:
        Dictionary containing:
        - score: Final combined reward (required)
        - Other keys: Sub-rewards that will be logged automatically
    """
    
    # 1. Fluency Score - based on response length being in optimal range
    response_length = len(solution_str.split())
    if response_length < min_length:
        fluency_score = response_length / min_length  # Penalize too short
    elif response_length > max_length:
        fluency_score = max_length / response_length  # Penalize too long  
    else:
        fluency_score = 1.0  # Optimal length
    
    # 2. Relevance Score - simplified keyword matching (placeholder)
    # In a real implementation, you might use semantic similarity, BERTScore, etc.
    # For demo, we'll check if the ground truth appears in the solution
    if ground_truth and ground_truth.lower() in solution_str.lower():
        relevance_score = 1.0
    else:
        relevance_score = 0.5  # Partial credit for demo
    
    # 3. Safety Score - placeholder for content safety
    # In a real implementation, you might use a safety classifier  
    # For demo, check for certain words that might indicate unsafe content
    unsafe_keywords = ["violence", "harm", "illegal"]
    if any(keyword in solution_str.lower() for keyword in unsafe_keywords):
        safety_score = 0.0
    else:
        safety_score = 1.0
    
    # 4. Combine into final reward
    final_score = (
        fluency_weight * fluency_score +
        relevance_weight * relevance_score + 
        safety_weight * safety_score
    )
    
    # Return dictionary with main score and sub-rewards
    return {
        "score": final_score,  # This is the main reward used for training
        "fluency_score": fluency_score,  # These will be logged as metrics
        "relevance_score": relevance_score,
        "safety_score": safety_score,
        "response_length": response_length,
        "data_source": data_source,  # Can include metadata too
    }


def simple_custom_reward(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    Simple custom reward function that only returns the score.
    This won't provide sub-reward logging but is simpler to implement.
    
    Args:
        data_source: The dataset name/identifier
        solution_str: The model's response/solution string
        ground_truth: The expected ground truth answer
        extra_info: Additional information
        
    Returns:
        float: The reward score
    """
    # Simple example: reward based on response length
    response_length = len(solution_str.split())
    return min(response_length / 50.0, 1.0)  # Normalize to 0-1 range


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    Default function name that VERL looks for if custom_reward_function.name is not specified.
    You can rename your main function to this if you want to avoid specifying the name in config.
    """
    return custom_reward_with_subrewards(data_source, solution_str, ground_truth, extra_info)
