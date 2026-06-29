import random
import os

file_dir = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(file_dir, "../data/shakespeare.txt")).read()


chars = sorted(set(text))
vocab_size = len(chars)  


stoi = {c: i for i, c in enumerate(chars)}  
itos = {i: c for c, i in stoi.items()}      

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

if __name__ == "__main__":
    print(f"Vocab size: {vocab_size}")
    print(f"Sample encode('Hello'): {encode('Hello')}")
    print(f"Sample decode back: {decode(encode('Hello'))}")