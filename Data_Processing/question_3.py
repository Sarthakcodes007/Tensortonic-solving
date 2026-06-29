seqs = [[1,2,3], [4,5], [6]]
# a = len(seqs[0])
# b = len(seqs[1])
# c = len(seqs[2])

# print (a, b, c)

# for seq in seqs:
#     print(len(seq))

# how to keep track of the highest one ? 

# max_len = 0 
# for seq in seqs:
#     if len(seq) > max_len:
#         max_len = len(seq)

# padding = max_len - len(seq)

# [0] * padding + seq
# my code attempt 1
# result = []

# max_len = 0

# for seq in seqs:
#     if max_len < len(seq):
#         max_len = len(seq)

# for seq in seqs:
#     padding = max_len - len(seq)
#     result.append([0] * padding + seq)

# print(result)

# the problem with this approach is that it will add the zeros to the left of the sequence, but we want to add them to the right.
# thus we will be using result.append(seq + [0] * padding) instead of result.append([0] * padding + seq)

import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    # Handle empty input
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)

    # Determine maximum length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    result = []

    for seq in seqs:
        # Truncate if longer than max_len
        seq = seq[:max_len]

        # Pad if shorter than max_len
        padding = max_len - len(seq)
        result.append(seq + [pad_value] * padding)

    return np.array(result, dtype=int)