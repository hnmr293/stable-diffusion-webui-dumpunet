import sys
import os
import time
import contextlib
from typing import Callable

import modules.scripts as scripts
from modules.processing import process_images, fix_seed, StableDiffusionProcessing, Processed

from scripts.lib.build_ui import UI
from scripts.lib.feature_extractor import FeatureExtractorBase
from scripts.lib.features.extractor import FeatureExtractor
from scripts.lib.features.utils import feature_diff, feature_to_grid_images
from scripts.lib.tutils import save_tensor
from scripts.lib.layer_prompt.prompt import LayerPrompt
from scripts.lib.attention.extractor import AttentionExtractor
from scripts.lib.report import message as E
from scripts.lib import putils

class Script(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.on_process: set[Callable] = set()
        self.on_process_batch: set[Callable] = set()
        self.debug = False
    
    def log(self, msg: str):
        if self.debug:
            print(E(msg), file=sys.stderr)
    
    def title(self):
        return "Dump U-Net features"
    
    def show(self, is_img2img):
        return True
    
    def ui(self, is_img2img):
        
        result: UI = UI.build(self, is_img2img)
        
        return [
            result.unet.enabled,
            result.unet.settings.layers,
            result.unet.settings.steps,
            result.unet.settings.color,
            result.unet.dump.enabled,
            result.unet.dump.path,
            
            result.attn.enabled,
            result.attn.settings.layers,
            result.attn.settings.steps,
            result.attn.settings.color,
            result.attn.dump.enabled,
            result.attn.dump.path,
            
            result.lp.enabled,
            result.lp.diff_enabled,
            result.lp.diff_settings.layers,
            result.lp.diff_settings.steps,
            result.lp.diff_settings.color,
            result.lp.diff_dump.enabled,
            result.lp.diff_dump.path,
            
            result.debug.log,
        ]
    
    def process(self, p, *args, **kwargs):
        for fn in self.on_process:
            fn(p, *args, **kwargs)
    
    def process_batch(self, p, *args, **kwargs):
        for fn in self.on_process_batch:
            fn(p, *args, **kwargs)
    
    def run(self,
            p: StableDiffusionProcessing,
            *args,
            **kwargs
    ):
        # Currently class scripts.Script does not support {post}process{_batch} hooks
        # for non-AlwaysVisible scripts.
        # So we have no legal method to access current batch number.
        
        # ugly hack
        if p.scripts is not None:
            p.scripts.alwayson_scripts.append(self)
            # now `process_batch` will be called from modules.processing.process_images
        
        try:
            return self.run_impl(p, *args, **kwargs)
        finally:
            if p.scripts is not None:
                p.scripts.alwayson_scripts.remove(self)
        
    def run_impl(self,
            p: StableDiffusionProcessing,
            
            unet_features_enabled: bool,
            layer_input: str,
            step_input: str,
            color: bool,
            path_on: bool,
            path: str,
            
            attn_enabled: bool,
            attn_layers: str,
            attn_steps: str,
            attn_color: bool,
            attn_path_on: bool,
            attn_path: str,
            
            layerprompt_enabled: bool,
            layerprompt_diff_enabled: bool,
            lp_diff_layers: str,
            lp_diff_steps: str,
            lp_diff_color: bool,
            diff_path_on: bool,
            diff_path: str,
            
            debug: bool,
    ):
                  
        if not unet_features_enabled and not attn_enabled and not layerprompt_enabled:
            return process_images(p)
        
        self.debug = debug
        
        ex = FeatureExtractor(
            self,
            unet_features_enabled,
            p.steps,
            layer_input,
            step_input,
            path if path_on else None
        )
        
        exlp = FeatureExtractor(
            self,
            layerprompt_diff_enabled,
            p.steps,
            lp_diff_layers,
            lp_diff_steps,
            path if path_on else None
        )
        
        lp = LayerPrompt(
            self,
            layerprompt_enabled,
        )
        
        at = AttentionExtractor(
            self,
            attn_enabled,
            p.steps,
            attn_layers,
            attn_steps,
            attn_path if attn_path_on else None
        )
        
        if layerprompt_diff_enabled:
            fix_seed(p)
            
            p1 = putils.copy(p)
            p2 = putils.copy(p)
            
            # layer prompt disabled
            lp0 = LayerPrompt(self, layerprompt_enabled, remove_layer_prompts=True)
            proc1 = exec(p1, lp0, [ex, exlp, at])
            features1 = ex.extracted_features
            diff1 = exlp.extracted_features
            proc1 = ex.add_images(p1, proc1, features1, color)
            proc1 = at.add_images(p1, proc1, at.extracted_features, attn_color)
            # layer prompt enabled
            proc2 = exec(p2, lp, [ex, exlp, at])
            features2 = ex.extracted_features
            diff2 = exlp.extracted_features
            proc2 = ex.add_images(p2, proc2, features2, color)
            proc2 = at.add_images(p2, proc2, at.extracted_features, attn_color)
            
            assert len(proc1.images) == len(proc2.images)
            
            proc = putils.merge(p, proc1, proc2)
            
            if diff_path_on:
                assert diff_path is not None and diff_path != "", E("<Output path> must not be empty.")
                # mkdir -p path
                if os.path.exists(diff_path):
                    assert os.path.isdir(diff_path), E("<Output path> already exists and is not a directory.")
                else:
                    os.makedirs(diff_path, exist_ok=True)
                
            t0 = int(time.time())
            for img_idx, step, layer, tensor in feature_diff(diff1, diff2, abs=not lp_diff_color):
                canvases = feature_to_grid_images(tensor, layer, p.width, p.height, lp_diff_color)
                for canvas in canvases:
                    putils.add_ref(proc, img_idx, canvas, f"Layer Name: {layer}, Feature Steps: {step}")
                    
                if diff_path_on:
                    basename = f"{img_idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                    save_tensor(tensor, diff_path, basename)
            
        else:
            proc = exec(p, lp, [ex, at])
            proc = ex.add_images(p, proc, ex.extracted_features, color)
            proc = at.add_images(p, proc, at.extracted_features, attn_color)
            
        return proc
    
    def notify_error(self, e: Exception):
        pass
    
    def set_debug(self, b: bool):
        self.debug = b

def exec(
    p: StableDiffusionProcessing,
    lp: LayerPrompt,
    extractors: list[FeatureExtractorBase]
):
    proc = None
    with lp:
        lp.setup(p)
        with contextlib.ExitStack() as ctx:
            for ex in extractors:
                ctx.enter_context(ex)
                ex.setup(p)
            proc = process_images(p)
    
    assert proc is not None
    return proc
