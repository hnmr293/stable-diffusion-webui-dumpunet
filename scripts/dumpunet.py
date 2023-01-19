import os
import time
import json

import gradio as gr

import modules.scripts as scripts
from modules.processing import process_images, fix_seed, StableDiffusionProcessing, Processed
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img

from scripts.dumpunet import layerinfo
from scripts.dumpunet.features.extractor import FeatureExtractor
from scripts.dumpunet.features.process import feature_diff, tensor_to_grid_images, save_tensor
from scripts.dumpunet.layer_prompt.prompt import LayerPrompt
from scripts.dumpunet.layer_prompt.parser import BadPromptError
from scripts.dumpunet.report import message as E

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
        
        ID_PREFIX = "dumpunet"
        def id(x):
            return f"{ID_PREFIX}-{'img2img' if is_img2img else 'txt2img'}-{x}"
        
        with gr.Group(elem_id=id("ui")):
            with gr.Tab("U-Net features", elem_id=id("features-tab")):
                unet_features_enabled = gr.Checkbox(
                    label="Extract U-Net features",
                    value=False,
                    elem_id=id("features-checkbox")
                )
                
                with gr.Group(elem_id=id("features-image")):
                    layer = gr.Textbox(
                        label="Layers",
                        placeholder="eg. IN00-OUT03(+2),OUT10",
                        elem_id=id("features-layer"),
                    )
                    
                    layer_setting_hidden = gr.HTML(
                        json.dumps(layerinfo.Settings),
                        visible=False,
                        elem_id=id("features-layer_setting")
                    )
                    
                    steps = gr.Textbox(
                        label="Image saving steps",
                        placeholder="eg. 1,5-20(+5)",
                        elem_id=id("features-steps")
                    )
                    
                    color = gr.Checkbox(
                        False,
                        label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)",
                        elem_id=id("features-color")
                    )
                    
                with gr.Group(elem_id=id("features-tensor")):
                    path_on = gr.Checkbox(
                        False,
                        label="Dump feature tensors to files",
                        elem_id=id("features-dump")
                    )
                    
                    path = gr.Textbox(
                        label="Output path",
                        placeholder="eg. /home/hnmr/unet/",
                        elem_id=id("features-dumppath")
                    )
            
                with gr.Accordion("Selected Layer Info", open=False):
                    layer_info = gr.HTML(elem_id=id("features-layerinfo"))
        
            with gr.Tab("Layer Prompt", elem_id=id("layerprompt-tab")):
                layerprompt_enabled = gr.Checkbox(
                    label="Enable Layer Prompt",
                    value=False,
                    elem_id=id("layerprompt-checkbox")
                )
                
                with gr.Group(elem_id=id("layerprompt-stdout")):
                    layerprompt_show_prompts = gr.Checkbox(
                        label="Show prompts in stdout",
                        value=False,
                        elem_id=id("layerprompt-stdout-checkbox")
                    )
                
                with gr.Group(elem_id=id("layerprompt-diff")):
                    layerprompt_diff_enabled = gr.Checkbox(
                        label="Output difference map of U-Net features between with and without Layer Prompt",
                        value=False,
                        elem_id=id("layerprompt-diff-checkbox")
                    )
                        
                    diff_path_on = gr.Checkbox(
                        False,
                        label="Dump difference tensors to files",
                        elem_id=id("layerprompt-diff-dump")
                    )
                    
                    diff_path = gr.Textbox(
                        label="Output path",
                        placeholder="eg. /home/hnmr/unet/",
                        elem_id=id("layerprompt-diff-dumppath")
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
            layerprompt_show_prompts,
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
            layerprompt_show_prompts: bool,
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
            unet_features_enabled or layerprompt_diff_enabled,
            p.steps,
            layer_input,
            step_input,
            path if path_on else None
        )
        
        lp = LayerPrompt(
            self,
            layerprompt_enabled,
            layerprompt_show_prompts,
        )
        
        if layerprompt_diff_enabled:
            fix_seed(p)
            
            p1 = copy_processing(p)
            p2 = copy_processing(p)
            
            # layer prompt disabled
            proc1, features1 = exec(p1, ex, lp, remove_layer_prompts=True)
            if unet_features_enabled:
                proc1 = ex.add_images(p1, proc1, features1, color)
            # layer prompt enabled
            proc2, features2 = exec(p2, ex, lp)
            if unet_features_enabled:
                proc2 = ex.add_images(p2, proc2, features2, color)
            
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
            
            if diff_path_on:
                assert diff_path is not None and diff_path != "", E("<Output path> must not be empty.")
                # mkdir -p path
                if os.path.exists(diff_path):
                    assert os.path.isdir(diff_path), E("<Output path> already exists and is not a directory.")
                else:
                    os.makedirs(diff_path, exist_ok=True)
                
            t0 = int(time.time())
            for img_idx, step, layer, tensor in feature_diff(features1, features2):
                canvases = tensor_to_grid_images(tensor, layer, p.width, p.height, color)
                for canvas in canvases:
                    proc.images.append(canvas)
                    proc.all_prompts.append(proc1.all_prompts[img_idx])
                    proc.all_negative_prompts.append(proc1.all_negative_prompts[img_idx])
                    proc.all_seeds.append(proc1.all_seeds[img_idx])
                    proc.all_subseeds.append(proc1.all_subseeds[img_idx])
                    proc.infotexts.append(f"Layer Name: {layer}, Feature Steps: {step}")
                    
                if diff_path_on:
                    basename = f"{img_idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                    save_tensor(tensor, diff_path, basename)
                    
            
        else:
            proc, features = exec(p, ex, lp)
            if unet_features_enabled:
                proc = ex.add_images(p, proc, features, color)
            
        return proc
    
    def notify_error(self, e: Exception):
        if isinstance(e, BadPromptError):
            if self.prompt_error is not None:
                self.prompt_error = str(e)

def exec(
    p: StableDiffusionProcessing,
    ex: FeatureExtractor,
    lp: LayerPrompt,
    remove_layer_prompts: bool = False
):
    with ex, lp:
        # replace U-Net forward, and...
        lp.setup(p, remove_layer_prompts=remove_layer_prompts)
        ex.setup(p) # hook the replaced forward
        # ex.__exit__ does clean up hooks
        
        proc = process_images(p)
        return proc, ex.extracted_features

def copy_processing(p: StableDiffusionProcessing):
    args = {
        "sd_model": p.sd_model,
        "outpath_samples": p.outpath_samples,
        "outpath_grids": p.outpath_grids,
        "prompt": p.prompt,
        "styles": p.styles[:],
        "seed": p.seed,
        "subseed": p.subseed,
        "subseed_strength": p.subseed_strength,
        "seed_resize_from_h": p.seed_resize_from_h,
        "seed_resize_from_w": p.seed_resize_from_w,
        "seed_enable_extras": False,
        "sampler_name": p.sampler_name,
        "batch_size": p.batch_size,
        "n_iter": p.n_iter,
        "steps": p.steps,
        "cfg_scale": p.cfg_scale,
        "width": p.width,
        "height": p.height,
        "restore_faces": p.restore_faces,
        "tiling": p.tiling,
        "do_not_save_samples": p.do_not_save_samples,
        "do_not_save_grid": p.do_not_save_grid,
        "extra_generation_params": p.extra_generation_params.copy() if p.extra_generation_params is not None else {},
        "overlay_images": p.overlay_images,
        "negative_prompt": p.negative_prompt,
        "eta": p.eta,
        "do_not_reload_embeddings": p.do_not_reload_embeddings,
        "denoising_strength": p.denoising_strength,
        "ddim_discretize": p.ddim_discretize,
        "s_churn": p.s_churn,
        "s_tmax": p.s_tmax,
        "s_tmin": p.s_tmin,
        "s_noise": p.s_noise,
        "override_settings": p.override_settings.copy() if p.override_settings is not None else {},
        "override_settings_restore_afterwards": p.override_settings_restore_afterwards,
        "sampler_index": None,
    }
    
    t2i_args = {}
    i2i_args = {}
    
    if isinstance(p, StableDiffusionProcessingTxt2Img):
        t2i_args = {
            "enable_hr": p.enable_hr,
            "denoising_strength": p.denoising_strength,
            "hr_scale": p.hr_scale,
            "hr_upscaler": p.hr_upscaler,
            "hr_second_pass_steps": p.hr_second_pass_steps,
            "hr_resize_x": p.hr_resize_x,
            "hr_resize_y": p.hr_resize_y,
        }
        if p.hr_upscale_to_x != 0 or p.hr_upscale_to_y != 0:
            t2i_args.update({
                "firstphase_width": p.width,
                "firstphase_height": p.height,
                "width": p.hr_upscale_to_x,
                "height": p.hr_upscale_to_y,
            })
    
    if isinstance(p, StableDiffusionProcessingImg2Img):
        i2i_args = {
            "init_images": p.init_images[:] if p.init_images is not None else None,
            "resize_mode": p.resize_mode,
            "denoising_strength": p.denoising_strength,
            "mask": p.image_mask,
            "mask_blur": p.mask_blur,
            "inpainting_fill": p.inpainting_fill,
            "inpaint_full_res": p.inpaint_full_res,
            "inpaint_full_res_padding": p.inpaint_full_res_padding,
            "inpainting_mask_invert": p.inpainting_mask_invert,
            "initial_noise_multiplier": p.initial_noise_multiplier
        }
    
    args.update(t2i_args)
    args.update(i2i_args)
    
    pp = type(p)(**args)
    
    pp.prompt_for_display = p.prompt_for_display
    pp.paste_to = p.paste_to # type: ignore
    pp.color_corrections = p.color_corrections # type: ignore
    pp.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
    pp.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
    pp.scripts = p.scripts
    pp.script_args = p.script_args
    pp.all_prompts = p.all_prompts
    pp.all_negative_prompts = p.all_negative_prompts
    pp.all_seeds = p.all_seeds
    pp.all_subseeds = p.all_subseeds
    
    for attr in [
        "sampler",
        "truncate_x", "truncate_y",
        "init_latent", "latent_mask", "mask_for_overlay", "mask", "nmask", "image_conditioning",
        
    ]:
        if hasattr(p, attr):
            setattr(pp, attr, getattr(p, attr))
    
    return pp
