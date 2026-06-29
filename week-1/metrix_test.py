from bigram_generator import bigrams, generate

sample_1 = generate(bigrams, seed=42,  length=200)
sample_2 = generate(bigrams, seed=123, length=200)
sample_3 = generate(bigrams, seed=999, length=200)

print("=== Seed 42 ===")
print(sample_1)
print("\n=== Seed 123 ===")
print(sample_2)
print("\n=== Seed 999 ===")
print(sample_3)


def compute_metrics(sample, name):
    length = len(sample)
    unique_chars = len(set(sample))
    

    repeat_count = sum(1 for i in range(1, len(sample)) if sample[i] == sample[i-1])
    repeat_ratio = repeat_count / (length - 1) if length > 1 else 0
    
    return {
        "Sample": name,
        "Length": length,
        "Unique Chars": unique_chars,
        "Repeated Char Ratio": round(repeat_ratio, 4)
    }

metrics = [
    compute_metrics(sample_1, "Seed 42"),
    compute_metrics(sample_2, "Seed 123"),
    compute_metrics(sample_3, "Seed 999"),
]


print(f"{'Sample':<12} {'Length':>8} {'Unique Chars':>13} {'Repeat Ratio':>13}")
print("-" * 50)
for m in metrics:
    print(f"{m['Sample']:<12} {m['Length']:>8} {m['Unique Chars']:>13} {m['Repeated Char Ratio']:>13}")