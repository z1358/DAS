from copy import deepcopy
from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY
import torch


@ADAPTATION_REGISTRY.register()
class DDA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

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
    
        # 判断是否为normalize_model包装
        # if isinstance(self.model, torch.nn.Sequential) and hasattr(self.model[0], "forward") and self.model[0].__class__.__name__ == "ImageNormalizer":
        #     outputs_test = self.model[1](imgs_test)
        #     outputs_syn = self.model[1](imgs_syn)  # 尝试跳过norm 效果变差非常多
        # else:
        #     outputs_test = self.model(imgs_test)
        #     outputs_syn = self.model(imgs_syn)

        outputs_test = self.model(imgs_test)
        outputs_syn = self.model(imgs_syn)
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
