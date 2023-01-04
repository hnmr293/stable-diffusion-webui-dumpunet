# Thanks for kohya.S
# https://note.com/kohya_ss/n/n93b7c01b0547

import sys
from typing import Callable, Any

import torch

from ldm.modules.diffusionmodules.util import timestep_embedding # type: ignore
from modules.processing import StableDiffusionProcessing
from modules import devices, prompt_parser

from scripts.dumpunet.layer_prompt.generator import LayerPromptGenerator, LayerPromptEraseGenerator, LayerPrompts
from scripts.dumpunet.layer_prompt.parser import BadPromptError

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
    
    def __init__(self, runner, enabled: bool):
        self._runner = runner
        self._enabled = enabled
        self.o_c = self.o_uc = ""
        self.o_fw = self.model = self.c = self.uc = self.fw = None

    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.model is not None and self.o_fw is not None:
            self.model.diffusion_model.forward = self.o_fw
        self.o_c = self.o_uc = ""
        self.o_fw = self.model = self.c = self.uc = self.fw = None
    
    def setup(self, p: StableDiffusionProcessing, disabled=False):
        if not self._enabled:
            return
        
        self.o_c = p.all_prompts[0] if len(p.all_prompts or []) > 0 else p.prompt # type: ignore
        self.o_uc = p.all_negative_prompts[0] if len(p.all_negative_prompts or []) > 0 else p.negative_prompt # type: ignore
        
        try:
            if disabled:
                gen = LayerPromptEraseGenerator()
            else:
                gen = LayerPromptGenerator()
            self.c = gen.generate(self.o_c)
            self.uc = gen.generate(self.o_uc)
        except BadPromptError as pe:
            self._runner.notify_error(pe)
            print("\033[31m", file=sys.stderr, end="", flush=False)
            print(pe.message(), file=sys.stderr, flush=False)
            print("\033[0m", file=sys.stderr, end="")
            raise ValueError(f"Prompt syntax error at pos={pe.pos}. See stderr for details.") from None
        
        print("=" * 80)
        print("Layer Prompts")
        print("-" * 80)
        for l, c in self.c.items():
            print(f"{l:<5} : {c}")
        print("-" * 80)
        print("Layer Negative Prompts")
        print("-" * 80)
        for l, c in self.uc.items():
            print(f"{l:<5} : {c}")
        print("=" * 80)
        
        old = p.sd_model.model.diffusion_model.forward # type: ignore
        new = self._forward_fn(p, self.c, self.uc)
        self.o_fw = old
        self.fw = new
        
        # replace U-Net forward
        self.model = p.sd_model.model                # type: ignore
        self.model.diffusion_model.forward = self.fw # type: ignore
    
    def _forward_fn(
        self,
        p: StableDiffusionProcessing,
        c: LayerPrompts,
        uc: LayerPrompts
    ):
        assert c is not None
        assert uc is not None
        
        blocks_cond = self._create_blocks_cond(p, c, uc)

        new_forward = self._create_forward_fn(
            p.sd_model.model.diffusion_model, # type: ignore
            blocks_cond
        )
        
        return new_forward
    
    def _create_blocks_cond(
        self,
        p: StableDiffusionProcessing,
        c: LayerPrompts,
        uc: LayerPrompts
    ):
        # get conditional
        with devices.autocast():
            uc1 = prompt_parser.get_learned_conditioning(p.sd_model, list(uc.values()), p.steps)
            c1 = prompt_parser.get_learned_conditioning(p.sd_model, list(c.values()), p.steps)
        
        blocks_cond = []
        for uc2, c2 in zip(uc1, c1):
            cond = torch.cat([c2[0].cond.unsqueeze(0), uc2[0].cond.unsqueeze(0)])                                         # ignore scheduled cond
            blocks_cond.append(cond)
        
        return blocks_cond
    
    def _create_forward_fn(self, _self, blocks_cond):
        
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
        
        return new_forward
    