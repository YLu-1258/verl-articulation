
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
