import os
import torch


def save_ckpt(logdir, model, prefix=''):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, os.path.join(logdir, prefix+'model.pth'))


def load_ckpt(logdir, model, prefix=''):
    save_pth = os.path.join(logdir, prefix+'model.pth')
    model.load_state_dict(torch.load(save_pth))
    return model