# Thanks for kohya.S
# https://note.com/kohya_ss/n/n93b7c01b0547

import sys
from collections import defaultdict
from typing import Callable, Any

import torch
from torch import Tensor

from ldm.modules.diffusionmodules.util import timestep_embedding # type: ignore
from modules.processing import StableDiffusionProcessing
from modules import devices, prompt_parser
from modules.prompt_parser import get_learned_conditioning as glc
from modules.prompt_parser import ScheduledPromptConditioning
glc : Callable[[Any,list[str],int],list[list[ScheduledPromptConditioning]]]

from scripts.lib import layerinfo
from scripts.lib.layer_prompt.generator import LayerPromptGenerator, LayerPromptEraseGenerator, LayerPrompts
from scripts.lib.layer_prompt.parser import BadPromptError

class LayerPrompt:
    
    # original forward function
    o_fw: Callable|None
    
    # hooker
    fw: Callable|None
    
    model: Any
    
    last_batch_num: int
    
    def __init__(self, runner, enabled: bool):
        self._runner = runner
        self._enabled = enabled
        self.o_fw = self.fw = self.model = None
        self.remove_layer_prompts = False
        self.last_batch_num = -1

    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None and self.o_fw is not None:
            self.model.diffusion_model.forward = self.o_fw
        self.o_fw = self.fw = self.model = None
        self.remove_layer_prompts = False
        self.last_batch_num = -1
    
    def setup(self, p: StableDiffusionProcessing, remove_layer_prompts=False):
        if not self._enabled:
            return
        
        self.remove_layer_prompts = remove_layer_prompts
        
        old = p.sd_model.model.diffusion_model.forward # type: ignore
        new = self._create_forward_fn(
            p.sd_model.model.diffusion_model, # type: ignore
            p
        )
        self.o_fw = old
        self.fw = new
        
        # replace U-Net forward
        self.model = p.sd_model.model                # type: ignore
        self.model.diffusion_model.forward = self.fw # type: ignore
    
    def _create_blocks_cond(
        self,
        p: StableDiffusionProcessing,
        all_prompts: list[str],
        all_negative_prompts: list[str]
    ):
        try:
            if self.remove_layer_prompts:
                gen = LayerPromptEraseGenerator()
            else:
                gen = LayerPromptGenerator()
            uc_list = [gen.generate(uc) for uc in all_negative_prompts]
            c_list = [gen.generate(c) for c in all_prompts]
        except BadPromptError as pe:
            self._runner.notify_error(pe)
            print("\033[31m", file=sys.stderr, end="", flush=False)
            print(pe.message(), file=sys.stderr, flush=False)
            print("\033[0m", file=sys.stderr, end="")
            raise ValueError(f"Prompt syntax error at pos={pe.pos}. See stderr for details.") from None
        
        # get conditional
        with devices.autocast():
            # ignore scheduled cond
            uc1 = [ glc(p.sd_model, list(image_uc.values()), p.steps) for image_uc in uc_list ]
            c1  = [ glc(p.sd_model, list(image_c.values()), p.steps)  for image_c  in c_list  ]
        
        steps = self._runner.steps_on_batch
        def get_cond(conds: list[ScheduledPromptConditioning]):
            assert len(conds) != 0
            for cc in conds:
                if steps <= cc.end_at_step:
                    return cc
            return conds[-1]
        
        # c,uc : img_idx -> layer_idx -> Tensor
        
        # layer_idx -> [0(c), 1(uc)] -> img_idx -> Tensor
        layer_conds : defaultdict[int,tuple[list[Tensor],list[Tensor]]]
        layer_conds = defaultdict(lambda: ([],[]))
        for img_idx, (uc2, c2) in enumerate(zip(uc1, c1)):
            assert len(uc2) == len(layerinfo.Names)
            assert len(c2) == len(layerinfo.Names)
            
            for layer_idx, (uc, c) in enumerate(zip(uc2, c2)):
                layer_conds[layer_idx][0].append(get_cond(c).cond)
                layer_conds[layer_idx][1].append(get_cond(uc).cond)
        
        # layer_idx -> [0(c), 1(uc)] -> Tensor
        layer_conds2 : list[tuple[Tensor,Tensor]]
        layer_conds2 = []
        for layer_idx in range(len(layerinfo.Names)):
            c, uc = layer_conds[layer_idx]
            layer_conds2.append(
                (
                    torch.cat([ cx.unsqueeze(0) for cx in c ]),
                    torch.cat([ ucx.unsqueeze(0) for ucx in uc ])
                )
            )
        
        # layer_idx -> Tensor
        layer_conds3 : list[Tensor]
        layer_conds3 = []
        for layer_idx in range(len(layerinfo.Names)):
            c, uc = layer_conds2[layer_idx]
            layer_conds3.append(torch.cat([c, uc]))
        
        return layer_conds3, c_list, uc_list
    
    def _create_forward_fn(self, _self, p):
        
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

            n = self._runner.batch_num - 1
            assert 0 <= n
            prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            
            assert len(prompts) != 0
            
            layer_conds, c_list, uc_list = \
                self._create_blocks_cond(p, prompts, negative_prompts)
            
            last_n, self.last_batch_num = self.last_batch_num, n
            if last_n != n:
                self.dump_prompts(n, c_list, uc_list)
            
            cond_index = 0
            h = x.type(_self.dtype)
            for module in _self.input_blocks:
                h = module(h, emb, layer_conds[cond_index] if context is not None else None)    # context)
                cond_index += 1
                hs.append(h)
            h = _self.middle_block(h, emb, layer_conds[cond_index] if context is not None else None)    # context)
            cond_index += 1
            for module in _self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, layer_conds[cond_index] if context is not None else None)    # context)
                cond_index += 1
            h = h.type(x.dtype)
            if _self.predict_codebook_ids:
                return _self.id_predictor(h)
            else:
                return _self.out(h)
        
        return new_forward
    
    def dump_prompts(self, n: int, c_list, uc_list):
        def pp(s): self._runner.log(s)
        pp("=" * 80)
        pp(f"Prompts (batch={n}, step={self._runner.steps_on_batch}")
        pp("-" * 80)
        for layername in layerinfo.Names:
            pp(f"{layername:<5} : {[getattr(c, layername) for c in c_list]}")
        pp("=" * 80)
        pp("Negative Prompts")
        pp("-" * 80)
        for layername in layerinfo.Names:
            pp(f"{layername:<5} : {[getattr(uc, layername) for uc in uc_list]}")
        pp("=" * 80)
