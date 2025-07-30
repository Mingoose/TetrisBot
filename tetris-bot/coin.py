import random

def flip_coins(probabilities):
    """
    Flips coins based on a list of probabilities for heads.

    Args:
        probabilities (list of float): A list of 12 probabilities, one for each coin.

    Returns:
        list of str: The result of each coin flip ('H' or 'T').
    """
    if len(probabilities) != 10:
        raise ValueError("You must provide exactly 12 probabilities.")

    results = []
    for i, p in enumerate(probabilities):
        if not (0 <= p <= 1):
            raise ValueError(f"Probability at index {i} is out of range: {p}")
        result = 'H' if random.random() < p else 'T'
        results.append(result)
    
    return results

# Example usage:
custom_probabilities = [(1/2), (1/3000), (1/3000), (1/3000), (1/5000), (1/5000), (1/5000), (1/9000), (1/9000), (1/9000)]
even_count = 0
for i in range(10000):
    flip_results = flip_coins(custom_probabilities)
    if flip_results.count('H') % 2 == 0:
        even_count += 1

print("Number of times an even number of heads was flipped:", even_count)

