import logging
import torch
import torch.nn as nn
import torch.jit
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    
@ADAPTATION_REGISTRY.register()
class REM(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.m = cfg.REM.M
        self.n = cfg.REM.N
        self.mn = [i * self.m for i in range(self.n)] # [0, 0.1, 0.2]
        self.lamb = cfg.REM.LAMB
        self.margin = cfg.REM.MARGIN * math.log(num_classes)

        self.entropy = Entropy()
        self.tokens = 196

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def loss_calculation(self, imgs_test):
        x = imgs_test[0]
        outputs, attn = self.model(x, return_attn=True)
        attn_score = attn.mean(dim=1)[:, 0, 1:] # attn shape [bs, 12, 197, 197]
        len_keeps = [] # attn_score [bs, 196]
        outputs_list = []

        self.model.eval()
        for m in self.mn:
            if m == 0.0:
                len_keeps.append(None)
                outputs_list.append(outputs)
            else:
                num_keep = int(self.tokens * (1 - m))
                len_keep = torch.topk(attn_score, num_keep, largest=False).indices
                len_keeps.append(len_keep)
                out = self.model(x, len_keep=len_keep, return_attn=False)
                outputs_list.append(out)
        self.model.train()

        loss = 0.0
        for i in range(1, len(self.mn)):
            loss += softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
            for j in range(1, i):
                loss += softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()

        entropys = [self.entropy(out) for out in outputs_list]
        lossn = 0.0
        for i in range(len(self.mn)):
            for j in range(i + 1, len(self.mn)):
                lossn += (F.relu(entropys[i] - entropys[j].detach() + self.margin)).mean()
            
        loss = loss + self.lamb * lossn

        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()    

        return outputs


    def configure_model(self):
        """Configure model."""
        self.model.train()
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        p.requires_grad = True
            else:
                m.requires_grad_(False)

    
    def collect_params_vit(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            # if 'layer4' in nm:
            #     continue
            # if 'blocks.9' in nm:
            #     continue
            # if 'blocks.10' in nm:
            #     continue
            # if 'blocks.11' in nm:
            #     continue
            # if 'norm.' in nm:
            #     continue
            # if nm in ['norm']:
            #     continue
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
        
@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)