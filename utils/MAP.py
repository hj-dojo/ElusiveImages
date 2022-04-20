import numpy as np
import pathlib

def compute_map(I, db, f, search_size):
  curr_precs = []
  match_count = 0.
  for i in range(1, search_size+1):
    if str(pathlib.Path(db.im_indices[I[0][i-1]]).parts[3]) != f:
      curr_precs.append(0)
    else:
      match_count += 1.
      curr_precs.append(match_count/i)
  curr_map = sum(curr_precs) / search_size
  return curr_map