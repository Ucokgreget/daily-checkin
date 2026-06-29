import random
from collections import defaultdict
from tokenizer_code import text, chars

def build_bigram_model(text):
    """Build bigram frequency table from text."""
    bigrams = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - 1):
        current_char = text[i]
        next_char = text[i + 1]
        bigrams[current_char][next_char] += 1
    return bigrams

def sample_next_char(bigrams, current_char, rng):
    """Sample next character given current character using seeded RNG."""
    if current_char not in bigrams:
        return rng.choice(chars)
    
    next_chars = list(bigrams[current_char].keys())
    counts = [bigrams[current_char][c] for c in next_chars]
    total = sum(counts)
    probs = [c / total for c in counts]
    
    r = rng.random()
    cumulative = 0.0
    for char, prob in zip(next_chars, probs):
        cumulative += prob
        if r < cumulative:
            return char
    return next_chars[-1]

def generate(bigrams, seed=42, start_char='\n', length=200):
    """Generate text using bigram model with a fixed seed."""
    rng = random.Random(seed)
    current = start_char
    result = [current]
    for _ in range(length - 1):
        current = sample_next_char(bigrams, current, rng)
        result.append(current)
    return "".join(result)

bigrams = build_bigram_model(text)