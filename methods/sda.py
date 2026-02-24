from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY
import torch


def load_checkpoint(model, checkpoint_path, map_location='cpu', needed_prefix=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # 兼容常见的 state_dict 格式
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # 统一去除常见前缀
    def remove_prefix(key):
        for prefix in ['backbone.', 'model.', 'head.', 'fc.', 'module.']:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = remove_prefix(k)
        if needed_prefix is not None:
            new_key = needed_prefix + new_key
        new_state_dict[new_key] = v

    # 尝试加载，允许部分不匹配
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
        # 处理 model.weight 和 model.bias
        if 'model.weight' in unexpected and 'model.bias' in unexpected:
            # 将其重定向到 model.head.weight 和 model.head.bias
            if 'model.weight' in new_state_dict:
                new_state_dict['model.head.weight'] = new_state_dict['model.weight']
            if 'model.bias' in new_state_dict:
                new_state_dict['model.head.bias'] = new_state_dict['model.bias']
            # 再次尝试加载
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"Missing keys after remap: {missing}")
            if unexpected:
                print(f"Unexpected keys after remap: {unexpected}")
    return model

@ADAPTATION_REGISTRY.register()
class SDA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        fine_tuned_model = deepcopy(model)
        fine_tuned_model = load_checkpoint(fine_tuned_model, cfg.SDA.FINE_TUNED_CHECKPOINT, map_location=self.device, needed_prefix='model.')
        fine_tuned_model = fine_tuned_model.to(next(self.model.parameters()).device)
        fine_tuned_model.requires_grad_(False)
        self.fine_tuned_model = fine_tuned_model
        self.fine_tuned_model.eval()

    @forward_decorator
    def forward_and_adapt(self, x):
        assert self.cfg.DATA_Paired, \
            "DDA method requires paired data (real and synthetic images). " \
            "Please set DATA_Paired to True in your configuration."
        x = x[0]
        imgs = torch.split(x, x.shape[0] // 2, dim=0) if isinstance(x, torch.Tensor) else x
        assert len(imgs) == 2, "Input should contain two sets of images: real and synthetic."
        imgs_test = imgs[0]
        imgs_syn = imgs[1]
    
        outputs_test = self.model(imgs_test)
        outputs_syn = self.fine_tuned_model(imgs_syn)
        outputs = {"ensemble": outputs_test + outputs_syn, "outputs_test": outputs_test, "outputs_syn": outputs_syn}
        return outputs

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
