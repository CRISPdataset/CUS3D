import torch.optim


def build_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop('type')
    optim = getattr(torch.optim, optim_type)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)

def build_inner_mlp_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "to_high_mlp" in name:
            # print('set hi')
            param.requires_grad = True
        if "to_low_mlp" in name:
            # print('set low')
            param.requires_grad = True
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop('type')
    optim = getattr(torch.optim, optim_type)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
