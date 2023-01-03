# Thanks for kohya.S
# https://note.com/kohya_ss/n/n93b7c01b0547

from typing import Callable, Any

import torch

from ldm.modules.diffusionmodules.util import timestep_embedding # type: ignore
from modules.processing import StableDiffusionProcessing
from modules import devices, prompt_parser

from scripts.dumpunet.layer_prompt.generator import LayerPromptGenerator, LayerPrompts

class LayerPrompt:
    
    # original prompt
    o_c: str
    
    # original negative prompt
    o_uc: str
    
    # original forward function
    o_fw: Callable|None
    
    model: Any
    
    # current prompt
    c: LayerPrompts|None
    
    # current negative prompt
    uc: LayerPrompts|None
    
    # hooker
    fw: Callable|None
    
    def __init__(self, enabled: bool):
        self._enabled = enabled
        self.o_c = self.o_uc = ""
        self.o_fw = self.model = self.c = self.uc = self.fw = None

    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None and self.o_fw is not None:
            self.model.diffusion_model.forward = self.o_fw
    
    def setup(self, p: StableDiffusionProcessing):
        if not self._enabled:
            return
        
        self.o_c = p.all_prompts[0] if len(p.all_prompts or []) > 0 else p.prompt # type: ignore
        self.o_uc = p.all_negative_prompts[0] if len(p.all_negative_prompts or []) > 0 else p.negative_prompt # type: ignore
        
        gen = LayerPromptGenerator()
        self.c = gen.generate(self.o_c)
        self.uc = gen.generate(self.o_uc)
        
        print("=" * 80)
        print("Layer Prompts")
        print("-" * 80)
        for l, c in self.c.items():
            print(l, ":", c)
        print("-" * 80)
        print("Layer Negative Prompts")
        print("-" * 80)
        for l, c in self.uc.items():
            print(l, ":", c)
        print("=" * 80)
        
        old, new = self._hook(p)
        self.o_fw = old
        self.fw = new
        
        self.model = p.sd_model.model                # type: ignore
        self.model.diffusion_model.forward = self.fw # type: ignore
    
    def _hook(self, p: StableDiffusionProcessing):
        assert self.c is not None
        assert self.uc is not None
        
        # get conditional
        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(p.sd_model, list(self.uc.values()), p.steps)
            c = prompt_parser.get_learned_conditioning(p.sd_model, list(self.c.values()), p.steps)
        
        blocks_cond = []
        for uc1, c1 in zip(uc, c):
            cond = torch.cat([c1[0].cond.unsqueeze(0), uc1[0].cond.unsqueeze(0)])                                         # ignore scheduled cond
            blocks_cond.append(cond)

        # replace U-Net forward
        model = p.sd_model.model # type: ignore
        org_forward = model.diffusion_model.forward
        _self = model.diffusion_model
        
        def new_forward(x, timesteps=None, context=None, y=None, **kwargs):
            """
            Apply the model to an input batch.
            :param x: an [N x C x ...] Tensor of inputs.
            :param timesteps: a 1-D batch of timesteps.
            :param context: conditioning plugged in via crossattn
            :param y: an [N] Tensor of labels, if class-conditional.
            :return: an [N x C x ...] Tensor of outputs.
            """
            # print("replaced", x.size(), timesteps, context.size() if context is not None else context)
            assert (y is not None) == (
                    _self.num_classes is not None
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, _self.model_channels, repeat_only=False)
            emb = _self.time_embed(t_emb)

            if _self.num_classes is not None:
                assert y is not None
                assert y.shape[0] == x.shape[0]
                emb = emb + _self.label_emb(y)

            cond_index = 0
            h = x.type(_self.dtype)
            for module in _self.input_blocks:
                h = module(h, emb, blocks_cond[cond_index] if context is not None else None)    # context)
                cond_index += 1
                hs.append(h)
            h = _self.middle_block(h, emb, blocks_cond[cond_index] if context is not None else None)    # context)
            cond_index += 1
            for module in _self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, blocks_cond[cond_index] if context is not None else None)    # context)
                cond_index += 1
            h = h.type(x.dtype)
            if _self.predict_codebook_ids:
                return _self.id_predictor(h)
            else:
                return _self.out(h)

        return org_forward, new_forward
    