import os
import torch
from shutil import copyfile

def load(save_path, graph, criterion_dict=None, optimizer=None, device=None):
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    graph.load_state_dict(state["graph"])
    if optimizer:
        optimizer.load_state_dict(state["optim"])
    if criterion_dict:
        for k in criterion_dict:
            criterion_dict[k].load_state_dict(state[k])
    return global_step

def save(save_path, graph, criterion_dict, optimizer, global_step, is_best=False):
    state = {
        "global_step": global_step,
        "graph": graph.state_dict(),
        "optim": optimizer.state_dict()
    }
    for k in criterion_dict:
        state[k] = criterion_dict[k].state_dict()
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path),
                                 "trained.pkg")
        copyfile(save_path, best_path)