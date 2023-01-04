import os
import json

import gradio as gr

import modules.scripts as scripts
from modules.processing import process_images, StableDiffusionProcessing
from modules import shared

from scripts.dumpunet import layerinfo
from scripts.dumpunet.features.feature_extractor import FeatureExtractor
from scripts.dumpunet.layer_prompt.prompt import LayerPrompt
from scripts.dumpunet.layer_prompt.parser import BadPromptError
from scripts.dumpunet.report import message as E
from scripts.dumpunet.utils import *

class Script(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.batch_num = 0
        self.steps_on_batch = 0
        self.prompt_error = ""
    
    def title(self):
        return "Dump U-Net features"
    
    def show(self, is_img2img):
        return True
    
    def ui(self, is_img2img):
        
        with gr.Group(elem_id="dumpunet-ui"):
            with gr.Tab("U-Net features", elem_id="dumpunet-features-tab"):
                unet_features_enabled = gr.Checkbox(
                    label="Extract U-Net features",
                    value=False,
                    elem_id="dumpunet-features-checkbox"
                )
                
                with gr.Group(elem_id="dumpunet-features-image"):
                    layer = gr.Textbox(
                        label="Layers",
                        placeholder="eg. IN00-OUT03(+2),OUT10",
                        elem_id="dumpunet-features-layer",
                    )
                    
                    layer_setting_hidden = gr.HTML(
                        json.dumps(layerinfo.Settings),
                        visible=False,
                        elem_id="dumpunet-features-layer_setting"
                    )
                    
                    steps = gr.Textbox(
                        label="Image saving steps",
                        placeholder="eg. 1,5-20(+5)",
                        elem_id="dumpunet-features-steps"
                    )
                    
                    color = gr.Checkbox(
                        False,
                        label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)",
                        elem_id="dumpunet-features-color"
                    )
                    
                with gr.Group(elem_id="dumpunet-features-tensor"):
                    path_on = gr.Checkbox(
                        False,
                        label="Dump feature tensors to files",
                        elem_id="dumpunet-features-dump"
                    )
                    
                    path = gr.Textbox(
                        label="Output path",
                        placeholder="eg. /home/hnmr/unet/",
                        elem_id="dumpunet-features-dumppath"
                    )
            
                with gr.Accordion("Selected Layer Info", open=False):
                    layer_info = gr.HTML(elem_id="dumpunet-features-layerinfo")
        
            with gr.Tab("Layer Prompt", elem_id="dumpunet-layerprompt-tab"):
            #with gr.Group(elem_id="dumpunet-layerprompt"):
                layerprompt_enabled = gr.Checkbox(
                    label="Enable Layer Prompt",
                    value=False,
                    elem_id="dumpunet-layerprompt-checkbox"
                )
                
                with gr.Group(elem_id="dumpunet-layerprompt-diff"):
                    layerprompt_diff_enabled = gr.Checkbox(
                        label="Output difference map of U-Net features between with and without Layer Prompt",
                        value=False,
                        elem_id="dumpunet-layerprompt-diff-checkbox"
                    )
                        
                    diff_path_on = gr.Checkbox(
                        False,
                        label="Dump difference tensors to files",
                        elem_id="dumpunet-layerprompt-diff-dump"
                    )
                    
                    diff_path = gr.Textbox(
                        label="Output path",
                        placeholder="eg. /home/hnmr/unet/",
                        elem_id="dumpunet-layerprompt-diff-dumppath"
                    )
                    
                    #with gr.Accordion("Prompt Errors", open=False):
                    #    prompt_error = gr.HTML(elem_id="dumpunet-layerprompt-errors")
                        
        return [
            unet_features_enabled,
            layer,
            steps,
            color,
            path_on,
            path,
            layerprompt_enabled,
            layerprompt_diff_enabled,
            diff_path_on,
            diff_path,
        ]
    
    def process(self, p, *args):
        self.batch_num = 0
    
    def process_batch(self, p, *args, **kwargs):
        self.steps_on_batch = 0
        self.batch_num += 1
    
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
            layerprompt_enabled: bool,
            layerprompt_diff_enabled: bool,
            diff_path_on: bool,
            diff_path: str,
    ):
                  
        self.prompt_error = ""
        
        if not unet_features_enabled and not layerprompt_enabled:
            return process_images(p)
        
        ex = FeatureExtractor(
            self,
            unet_features_enabled,
            p.steps,
            layer_input,
            step_input,
            path if path_on else None
        )
        
        lp = LayerPrompt(
            self,
            layerprompt_enabled,
        )
        
        #if layerprompt_diff_enabled:
        #    
        #    # ...
        #    
        #    if diff_path_on:
        #        assert diff_path is not None and diff_path != "", E("<Output path> must not be empty.")
        #        # mkdir -p path
        #        if os.path.exists(diff_path):
        #            assert os.path.isdir(diff_path), E("<Output path> already exists and is not a directory.")
        #        else:
        #            os.makedirs(diff_path, exist_ok=True)
        #        
        #        # ...
                    
        
        with ex, lp:
            lp.setup(p) # replace U-Net forward, and...
            ex.setup(p) # hook the replaced forward
            proc = process_images(p)
            # ex.__exit__ does clean up hooks
        
        if shared.state.interrupted:
            return proc
        
        proc = ex.create_feature_map(p, proc, color)
        
        return proc
    
    def notify_error(self, e: Exception):
        if isinstance(e, BadPromptError):
            if self.prompt_error is not None:
                self.prompt_error = str(e)
