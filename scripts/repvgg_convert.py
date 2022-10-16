import torch
import copy


def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == '__main__':
    from models.model_repvggv4 import Space_RepVGG

    model = Space_RepVGG(deploy=False)

    model.load_state_dict(torch.load(r'E:\work\pytorch_training_tool_seg\Space_seg\run\Seg_Space_RepVGGv4_2022-05-12-22-28\ckpt.pth')['model'])

    repvgg_model_convert(model, save_path='repvgg_converted.pth')
