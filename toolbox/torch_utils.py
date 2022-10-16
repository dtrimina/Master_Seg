import torch


def load_ckeckpoint(model, pth_path):
    pretrained_dict = torch.load(pth_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)