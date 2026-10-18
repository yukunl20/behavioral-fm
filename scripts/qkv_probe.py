import math
import torch

torch.set_printoptions(precision=4, sci_mode=False) # printing format

q = torch.tensor([[2.0, 0.0]]) # query [1, d_k]
k = torch.tensor([[1.0, 0.0],   # key 1 [d_k]; aligned with query
                  [0.0, 1.0]]) # key 2 [d_k]; orthogonal to query
v = torch.tensor([[0.0, 3.0],  # val 1 [d_v]; content mostly in second dimension
                  [4.0, 0.0]]) # val 2 [d_v]; content mostly in first dimension


def attend(q, k, v):
    logits = q @ k.T / math.sqrt(k.shape[-1])   # compute scaled dot-product logits [1, 2]
    weight = logits.softmax(dim=-1)             # logits into attention weights [1, 2]
    output = weight @ v                         # weighted sum of values, output [1, d_v]
    return logits, weight, output

for name, key, value in [              # Run controlled variants.
    ("base", k, v),                    # Original keys and values.
    ("larger_key_norm", 3.0 * k, v),   # Scale keys; changes logits and sharpness.
    ("larger_value_norm", k, 3.0 * v), # Scale values; weights stay same but output magnitude changes.
    ("swap_values", k, v.flip(0)),     # Keep addresses same but swap transported content.
]:
    logits, weight, output = attend(q, key, value) # Run attention for this variant.
    print("\n", name)                 # Print variant name.
    print("logits", logits)           # Print raw attention scores.
    print("weights", weight)          # Print attention weights after softmax.
    print("output", output)           # Print mixed value output.



