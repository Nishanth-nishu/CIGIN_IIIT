import numpy as np
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_len_matrix(len_list):
    """
    Create length matrix for batch processing with proper error handling
    """
    try:
        if not len_list or len(len_list) == 0:
            return torch.zeros((1, 1), dtype=torch.float32)
        
        len_list = np.array(len_list)
        max_nodes = int(np.sum(len_list))
        
        if max_nodes == 0:
            return torch.zeros((len(len_list), 1), dtype=torch.float32)
        
        curr_sum = 0
        len_matrix = []
        
        for l in len_list:
            l = int(l)  # Ensure integer
            curr = np.zeros(max_nodes, dtype=np.float32)
            if l > 0:  # Only set if length is positive
                end_idx = min(curr_sum + l, max_nodes)  # Prevent overflow
                curr[curr_sum:end_idx] = 1
            len_matrix.append(curr)
            curr_sum += l
        
        result = torch.tensor(np.array(len_matrix), dtype=torch.float32)
        
        # Ensure minimum dimensions
        if result.shape[0] == 0:
            result = torch.zeros((1, max_nodes), dtype=torch.float32)
        if result.shape[1] == 0:
            result = torch.zeros((result.shape[0], 1), dtype=torch.float32)
            
        return result
        
    except Exception as e:
        print(f"Error in get_len_matrix: {e}")
        print(f"len_list: {len_list}")
        # Return safe default
        batch_size = len(len_list) if len_list else 1
        return torch.zeros((batch_size, 1), dtype=torch.float32)