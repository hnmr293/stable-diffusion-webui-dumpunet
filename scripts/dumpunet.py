import sys
import os
import time
import contextlib
from typing import Callable

import modules.scripts as scripts
from modules.processing import process_images, fix_seed, StableDiffusionProcessing, Processed

from scripts.dumpunetlib.build_ui import UI
from scripts.dumpunetlib.feature_extractor import FeatureExtractorBase
from scripts.dumpunetlib.features.extractor import FeatureExtractor
from scripts.dumpunetlib.features.utils import feature_diff, feature_to_grid_images
from scripts.dumpunetlib.tutils import save_tensor
from scripts.dumpunetlib.putils import ProcessedBuilder
from scripts.dumpunetlib.colorizer import Colorizer
from scripts.dumpunetlib.layer_prompt.prompt import LayerPrompt
from scripts.dumpunetlib.attention.extractor import AttentionExtractor
from scripts.dumpunetlib.report import message as E
from scripts.dumpunetlib import putils

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
            result.unet.settings.average,
            result.unet.settings.colorize, result.unet.settings.colorspace,
            result.unet.settings.R, result.unet.settings.G, result.unet.settings.B,
            result.unet.settings.H, result.unet.settings.S, result.unet.settings.L,
            result.unet.settings.trans,
            result.unet.settings.linear_min, result.unet.settings.linear_max,
            result.unet.settings.sigmoid_gain, result.unet.settings.sigmoid_offset,
            result.unet.dump.enabled,
            result.unet.dump.path,
            
            result.attn.enabled,
            result.attn.settings.layers,
            result.attn.settings.steps,
            result.attn.settings.average,
            result.attn.settings.others["vqks"],
            result.attn.settings.colorize, result.attn.settings.colorspace,
            result.attn.settings.R, result.attn.settings.G, result.attn.settings.B,
            result.attn.settings.H, result.attn.settings.S, result.attn.settings.L,
            result.attn.settings.trans,
            result.attn.settings.linear_min, result.attn.settings.linear_max,
            result.attn.settings.sigmoid_gain, result.attn.settings.sigmoid_offset,
            result.attn.dump.enabled,
            result.attn.dump.path,
            
            result.lp.enabled,
            result.lp.diff_enabled,
            result.lp.diff_settings.layers,
            result.lp.diff_settings.steps,
            result.lp.diff_settings.average,
            result.lp.diff_settings.colorize, result.lp.diff_settings.colorspace,
            result.lp.diff_settings.R, result.lp.diff_settings.G, result.lp.diff_settings.B,
            result.lp.diff_settings.H, result.lp.diff_settings.S, result.lp.diff_settings.L,
            result.lp.diff_settings.trans,
            result.lp.diff_settings.linear_min, result.lp.diff_settings.linear_max,
            result.lp.diff_settings.sigmoid_gain, result.lp.diff_settings.sigmoid_offset,
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
            favg: bool,
            color_: str, colorspace: str,
            fr: str, fg: str, fb: str,
            fh: str, fs: str, fl: str,
            ftrans: str, flmin: float, flmax: float, fsig_gain: float, fsig_offset: float,
            path_on: bool,
            path: str,
            
            attn_enabled: bool,
            attn_layers: str,
            attn_steps: str,
            aavg: bool,
            attn_vqks: list[str],
            attn_color_: str, attn_cs: str,
            ar: str, ag: str, ab: str,
            ah: str, as_: str, al: str,
            atrans: str, almin: float, almax: float, asig_gain: float, asig_offset: float,
            attn_path_on: bool,
            attn_path: str,
            
            layerprompt_enabled: bool,
            layerprompt_diff_enabled: bool,
            lp_diff_layers: str,
            lp_diff_steps: str,
            lavg: bool,
            lp_diff_color_: str, lcs: str,
            lr: str, lg: str, lb: str,
            lh: str, ls: str, ll: str,
            ltrans: str, llmin: float, llmax: float, lsig_gain: float, lsig_offset: float,
            diff_path_on: bool,
            diff_path: str,
            
            debug: bool,
    ):
        
        if not unet_features_enabled and not attn_enabled and not layerprompt_enabled:
            return process_images(p)
        
        self.debug = debug
        
        color =         Colorizer(color_, colorspace,   (fr, fg, fb), (fh, fs, fl),  ftrans, (flmin, flmax), (fsig_gain, fsig_offset))
        attn_color =    Colorizer(attn_color_, attn_cs, (ar, ag, ab), (ah, as_, al), atrans, (almin, almax), (asig_gain, asig_offset))
        lp_diff_color = Colorizer(lp_diff_color_, lcs,  (lr, lg, lb), (lh, ls, ll) , ltrans, (llmin, llmax), (lsig_gain, lsig_offset))
        
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
            attn_vqks,
            attn_path if attn_path_on else None
        )
        
        if layerprompt_enabled and layerprompt_diff_enabled:
            fix_seed(p)
            
            p1 = putils.copy(p)
            p2 = putils.copy(p)
            
            # layer prompt disabled
            lp0 = LayerPrompt(self, layerprompt_enabled, remove_layer_prompts=True)
            proc1, features1, diff1, attn1 = exec(p1, lp0, [ex, exlp, at])
            builder1 = ProcessedBuilder()
            builder1.add_proc(proc1)
            ex.add_images(p1, builder1, features1, favg, color)
            at.add_images(p1, builder1, attn1, aavg, attn_color)
            # layer prompt enabled
            proc2, features2, diff2, attn2 = exec(p2, lp, [ex, exlp, at])
            builder2 = ProcessedBuilder()
            builder2.add_proc(proc1)
            ex.add_images(p2, builder2, features2, favg, color)
            at.add_images(p2, builder2, attn2, aavg, attn_color)
            
            proc1 = builder1.to_proc(p1, proc1)
            proc2 = builder2.to_proc(p2, proc2)
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
                canvases = feature_to_grid_images(tensor, layer, p.width, p.height, lavg, lp_diff_color)
                for canvas in canvases:
                    putils.add_ref(proc, img_idx, canvas, f"Layer Name: {layer}, Feature Steps: {step}")
                    
                if diff_path_on:
                    basename = f"{img_idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                    save_tensor(tensor, diff_path, basename)
            
        else:
            proc, features1, attn1 = exec(p, lp, [ex, at])
            builder = ProcessedBuilder()
            builder.add_proc(proc)
            ex.add_images(p, builder, features1, favg, color)
            at.add_images(p, builder, attn1, aavg, attn_color)
            proc = builder.to_proc(p, proc)
            
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
    return proc, *[ex.extracted_features for ex in extractors]
