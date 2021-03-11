from typing import List
import torch 
def cls_stw(txt: str):
    '''
    clean stop words
    '''    
    pass 


def trim_sequence(features, max_len: int):
    '''
    trim the sequence
    '''
    if len(features[0]) > max_len:
        tmp = []
        for i in features:
            tmp.append(i[:max_len])
        # print("out of range tensor")
        # print(torch.stack(tmp))
        return torch.stack(tmp)
    else:
        return features
            
# x = torch.tensor([[1 for i in range(10)],
#                  [2 for i in range(10)],
#                  [3 for i in range(10)]])

# print(x)
# print(trim_sequence(x, 5))

