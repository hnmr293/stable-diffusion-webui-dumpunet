from functools import reduce

from modules.processing import StableDiffusionProcessing
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.processing import StableDiffusionProcessingImg2Img
from modules.processing import Processed

def copy(p: StableDiffusionProcessing):
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

def merge(p: StableDiffusionProcessing, *procs: Processed) -> Processed:
    if len(procs) == 0:
        return Processed(p, [])
    
    return Processed(
        p,
        sum([pp.images[:pp.index_of_first_image] for pp in procs], start=[]) +
        sum([pp.images[pp.index_of_first_image:] for pp in procs], start=[]),
        procs[0].seed,
        procs[0].info,
        procs[0].subseed,
        [pp.all_prompts for pp in procs],
        [pp.all_negative_prompts for pp in procs],
        [pp.all_seeds for pp in procs],
        [pp.all_subseeds for pp in procs],
        sum([pp.index_of_first_image for pp in procs]),
        sum([pp.infotexts[:pp.index_of_first_image] for pp in procs], start=[]) +
        sum([pp.infotexts[pp.index_of_first_image:] for pp in procs], start=[]),
        "\n".join([pp.comments for pp in procs])
    )

def add_ref(proc: Processed, ref_idx: int, image, infotext: str = ""):
    proc.images.append(image)
    proc.all_prompts.append(proc.all_prompts[ref_idx])
    proc.all_negative_prompts.append(proc.all_negative_prompts[ref_idx])
    proc.all_seeds.append(proc.all_seeds[ref_idx])
    proc.all_subseeds.append(proc.all_subseeds[ref_idx])
    proc.infotexts.append(infotext)
