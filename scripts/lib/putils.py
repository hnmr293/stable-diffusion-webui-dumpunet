from dataclasses import dataclass
from collections import defaultdict
from typing import Any

from PIL.Image import Image

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

@dataclass
class ProcessedItem:
    image: Image
    seed: int
    subseed: int
    prompt: str
    negative_prompt: str
    infotext: str

@dataclass
class ProcessedItemRef:
    ref_idx: int
    image: Image|None
    infotext: str|None

class ProcessedBuilder:
    
    previews: list[tuple[Image,str]]
    
    items: list[ProcessedItem]
    
    # `index of self.items` -> ProcessedItemRef
    ref_items: defaultdict[int, list[ProcessedItemRef]]
    
    def __init__(self):
        self.previews = []
        self.items = []
        self.ref_items = defaultdict(lambda: [])

    def add_preview(self, image: Image, infotext: str, additional_info: dict = {}):
        if len(additional_info) != 0:
            if 0 < len(infotext):
                infotext += "\n"
            infotext += ", ".join([f"{k}: {v}" for k, v in additional_info.items()])
        
        self.previews.append((image, infotext))
    
    def add(self, image: Image, seed: int, subseed: int|None, prompt: str, neg_prompt: str, infotext: str, additional_info: dict = {}):
        if subseed is None:
            subseed = -1
        
        if len(additional_info) != 0:
            if 0 < len(infotext):
                infotext += "\n"
            infotext += ", ".join([f"{k}: {v}" for k, v in additional_info.items()])
        
        self.items.append(ProcessedItem(
            image,
            seed,
            subseed,
            prompt,
            neg_prompt,
            infotext
        ))
    
    def add_ref(self, ref_idx: int, image: Image|None, infotext: str|None, additional_info: dict = {}):
        if (not 0 <= ref_idx < len(self.items)) and (not -len(self.items) < ref_idx <= -1):
            raise IndexError(f"given={ref_idx}, #items={len(self.items)}")
        
        if ref_idx < 0:
            ref_idx += len(self.items)
        
        if len(additional_info) != 0:
            if infotext is None:
                infotext = ""
            if 0 < len(infotext):
                infotext += "\n"
            infotext += ", ".join([f"{k}: {v}" for k, v in additional_info.items()])
        
        self.ref_items[ref_idx].append(ProcessedItemRef(
            ref_idx,
            image,
            infotext
        ))
    
    def add_proc(self, proc: Processed):
        # For Dynamic Prompt Extension
        # which is not append subseeds...
        subseeds = proc.all_subseeds.copy()
        while len(subseeds) < len(proc.all_seeds):
            subseeds.append(subseeds[0] if 0 < len(subseeds) else -1)
        
        preview, rest = proc.images[:proc.index_of_first_image], proc.images[proc.index_of_first_image:]
        txt, txt_rest = proc.infotexts[:proc.index_of_first_image], proc.infotexts[proc.index_of_first_image:]
        for img, t in zip(preview, txt):
            self.add_preview(img, t)
        
        def get(L: list, idx: int, default: Any = None):
            if idx < len(L):
                return L[idx]
            elif 0 < len(L):
                return L[-1]
            else:
                return default
        
        for idx, (img, txt) in enumerate(zip(rest, txt_rest)):
            self.add(
                img,
                get(proc.all_seeds, idx, -1),
                get(proc.all_subseeds, idx, None),
                get(proc.all_prompts, idx, ""),
                get(proc.all_negative_prompts, idx, ""),
                txt
            )
    
    def to_proc(self, p: StableDiffusionProcessing, base_proc: Processed):
        items = self._fix_all()
        return Processed(
            p,
            [image for image, infotext in self.previews] + [item.image for item in items],
            seed=base_proc.seed,
            info=base_proc.info,
            subseed=base_proc.subseed,
            all_seeds=[item.seed for item in items],
            all_subseeds=[item.subseed for item in items],
            all_prompts=[item.prompt for item in items],
            all_negative_prompts=[item.negative_prompt for item in items],
            infotexts=[infotext for image, infotext in self.previews] + [item.infotext for item in items],
            index_of_first_image=len(self.previews)
        )
    
    def _fix(self, ref: ProcessedItemRef):
        assert 0 <= ref.ref_idx < len(self.items)
        item = self.items[ref.ref_idx]
        return ProcessedItem(
            ref.image if ref.image is not None else item.image,
            item.seed,
            item.subseed,
            item.prompt,
            item.negative_prompt,
            ref.infotext if ref.infotext is not None else item.infotext
        )
    
    def _fix_all(self):
        result: list[ProcessedItem] = []
        for idx, item in enumerate(self.items):
            refs = self.ref_items.get(idx, [])
            fixed = [self._fix(ref) for ref in refs]
            result.append(item)
            result.extend(fixed)
        return result
            
