import time
import json

import gradio as gr

import modules.scripts as scripts
from modules.processing import process_images, fix_seed, StableDiffusionProcessing, Processed
from modules import shared

from scripts.dumpunet import layerinfo
from scripts.dumpunet.features.extractor import FeatureExtractor
from scripts.dumpunet.features.process import feature_diff, tensor_to_grid_images
from scripts.dumpunet.layer_prompt.prompt import LayerPrompt
from scripts.dumpunet.layer_prompt.parser import BadPromptError

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
        
        layerprompt_diff_enabled = layerprompt_enabled and layerprompt_diff_enabled
        
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
        
        if layerprompt_diff_enabled:
            fix_seed(p)
            
            # layer prompt disabled
            proc1, features1 = exec(p, ex, lp, color, False)
            # layer prompt enabled
            proc2, features2 = exec(p, ex, lp, color)
            
            assert len(proc1.images) == len(proc2.images)
            
            proc = Processed(
                p,
                (
                    proc1.images[:proc1.index_of_first_image] + 
                    proc2.images[:proc2.index_of_first_image] + 
                    proc1.images[proc1.index_of_first_image:] + 
                    proc2.images[proc2.index_of_first_image:]
                ),
                proc1.seed,
                proc1.info,
                proc1.subseed,
                proc1.all_prompts + proc2.all_prompts,
                proc1.all_negative_prompts + proc2.all_negative_prompts,
                proc1.all_seeds + proc2.all_seeds,
                proc1.all_subseeds + proc2.all_subseeds,
                proc1.index_of_first_image + proc2.index_of_first_image,
                (
                    proc1.infotexts[:proc1.index_of_first_image] +
                    proc2.infotexts[:proc2.index_of_first_image] +
                    proc1.infotexts[proc1.index_of_first_image:] +
                    proc2.infotexts[proc2.index_of_first_image:]
                ),
                proc1.comments
            )
            
            #t0 = int(time.time())
            #basename = f"{img_idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
            for img_idx, step, layer, tensor in feature_diff(features1, features2):
                canvases = tensor_to_grid_images(tensor, layer, p.width, p.height, color)
                for canvas in canvases:
                    proc.images.append(canvas)
                    proc.all_prompts.append(proc1.all_prompts[img_idx])
                    proc.all_negative_prompts.append(proc1.all_negative_prompts[img_idx])
                    proc.all_seeds.append(proc1.all_seeds[img_idx])
                    proc.all_subseeds.append(proc1.all_subseeds[img_idx])
                    proc.infotexts.append(f"Layer Name: {layer}, Feature Steps: {step}")
            
            #if diff_path_on:
            #    assert diff_path is not None and diff_path != "", E("<Output path> must not be empty.")
            #    # mkdir -p path
            #    if os.path.exists(diff_path):
            #        assert os.path.isdir(diff_path), E("<Output path> already exists and is not a directory.")
            #    else:
            #        os.makedirs(diff_path, exist_ok=True)
            #    
            #    # ...
            
        else:
            proc, _ = exec(p, ex, lp, color)
            
        return proc
    
    def notify_error(self, e: Exception):
        if isinstance(e, BadPromptError):
            if self.prompt_error is not None:
                self.prompt_error = str(e)

def exec(
    p: StableDiffusionProcessing,
    ex: FeatureExtractor,
    lp: LayerPrompt,
    color: bool,
    enabled: bool = True
):
    with ex, lp:
        # replace U-Net forward, and...
        lp.setup(p, disabled=not enabled)
        ex.setup(p) # hook the replaced forward
        # ex.__exit__ does clean up hooks
        
        proc = process_images(p)
        proc = ex.create_feature_map(p, proc, color)
        return proc, ex.extracted_features
