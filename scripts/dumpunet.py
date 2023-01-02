import os
import time
import json
import re
import math
from collections import defaultdict

import numpy as np
from torch import nn, Tensor
import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import process_images, Processed, StableDiffusionProcessing
from modules import shared

LayerSettings = {
    #            input shape   output shape
    "IN00":   ( (   4, 8, 8), ( 320, 8, 8) ),
    "IN01":   ( ( 320, 8, 8), ( 320, 8, 8) ),
    "IN02":   ( ( 320, 8, 8), ( 320, 8, 8) ),
    "IN03":   ( ( 320, 8, 8), ( 320, 4, 4) ),
    "IN04":   ( ( 320, 4, 4), ( 640, 4, 4) ),
    "IN05":   ( ( 640, 4, 4), ( 640, 4, 4) ),
    "IN06":   ( ( 640, 4, 4), ( 640, 2, 2) ),
    "IN07":   ( ( 640, 2, 2), (1280, 2, 2) ),
    "IN08":   ( (1280, 2, 2), (1280, 2, 2) ),
    "IN09":   ( (1280, 2, 2), (1280, 1, 1) ),
    "IN10":   ( (1280, 1, 1), (1280, 1, 1) ),
    "IN11":   ( (1280, 1, 1), (1280, 1, 1) ),
    "M00":    ( (1280, 1, 1), (1280, 1, 1) ),
    "OUT00":  ( (2560, 1, 1), (1280, 1, 1) ),
    "OUT01":  ( (2560, 1, 1), (1280, 1, 1) ),
    "OUT02":  ( (2560, 1, 1), (1280, 2, 2) ),
    "OUT03":  ( (2560, 2, 2), (1280, 2, 2) ),
    "OUT04":  ( (2560, 2, 2), (1280, 2, 2) ),
    "OUT05":  ( (1920, 2, 2), (1280, 4, 4) ),
    "OUT06":  ( (1920, 4, 4), ( 640, 4, 4) ),
    "OUT07":  ( (1280, 4, 4), ( 640, 4, 4) ),
    "OUT08":  ( ( 960, 4, 4), ( 640, 8, 8) ),
    "OUT09":  ( ( 960, 8, 8), ( 320, 8, 8) ),
    "OUT10":  ( ( 640, 8, 8), ( 320, 8, 8) ),
    "OUT11":  ( ( 640, 8, 8), ( 320, 8, 8) ),
}

class Script(scripts.Script):
    
    def __init__(self) -> None:
        super().__init__()
        self.steps_on_batch = 0
    
    def title(self):
        return "Dump U-net features"
    
    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        with gr.Blocks(elem_id="dumpunet"):
            layer = gr.Dropdown([f"IN{i:02}" for i in range(12)] + ["M00"] + [f"OUT{i:02}" for i in range(12)], label="Layer", value="M00", elem_id="dumpunet-layer")
            layer_setting_hidden = gr.HTML(json.dumps(LayerSettings), visible=False, elem_id="dumpunet-layer_setting")
            steps = gr.Textbox(label="Image saving steps")
            color = gr.Checkbox(False, label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)") 
            with gr.Blocks():
                path_on = gr.Checkbox(False, label="Dump tensor to files")
                path = gr.Textbox(label="Output path")
            layer_info = gr.HTML(elem_id="dumpunet-layerinfo")
        
        return [layer, steps, color, path_on, path]
    
    def process_batch(self, p, *args, **kwargs):
        self.steps_on_batch = 0
    
    def run(self,
            p: StableDiffusionProcessing,
            layer: str,
            step_input: str,
            color: bool,
            path_on: bool,
            path: str):
        
        # Currently class scripts.Script does not support {post}process{_batch} hooks
        # for non-AlwaysVisible scripts.
        # So we have no legal method to access current batch number.
        
        # ugly hack
        if p.scripts is not None:
            p.scripts.alwayson_scripts.append(self)
            # now `process_batch` will be called from modules.processing.process_images
        
        try:
            return self.run_impl(p, layer,step_input, color, path_on, path)
        finally:
            if p.scripts is not None:
                p.scripts.alwayson_scripts.remove(self)
    
    def run_impl(self,
                 p: StableDiffusionProcessing,
                 layer: str,
                 step_input: str,
                 color: bool,
                 path_on: bool,
                 path: str):
        
        IN = [ f"IN{i:02}" for i in range(12) ]
        OUT = [ f"OUT{i:02}" for i in range(12) ]
        
        assert layer is not None and layer != "", "[DumpUnet] <Layer> must not be empty."
        if path_on:
            assert path is not None and path != "", "[DumpUnet] <Output path> must not be empty."
        
        steps = retrieve_steps(step_input)
        grid_x, grid_y = get_grid_num(layer, p.width, p.height)
        
        unet = p.sd_model.model.diffusion_model # type: ignore
        
        #time_embed :  nn.modules.container.Sequential
        #input_blocks  : nn.modules.container.ModuleList
        #middle_block : ldm.modules.diffusionmodules.openaimodel.TimestepEmbedSequential
        #output_blocks : nn.modules.container.ModuleList
        #time_embed = unet.time_embed
        #input_blocks = unet.input_blocks
        #middle_block = unet.middle_block
        #output_blocks = unet.output_blocks
        #summary(unet, (4, 512, 512))
        
        # mkdir -p path
        if path_on:
            if os.path.exists(path):
                assert os.path.isdir(path), "[DumpUnet] <Output path> already exists and is not a directory."
            else:
                os.makedirs(path, exist_ok=True)
        
        target : nn.modules.Module
        if layer in IN:
            idx = IN.index(layer)
            target = unet.input_blocks[idx]
        elif layer == "M00":
            target = unet.middle_block
        elif layer in OUT:
            idx = OUT.index(layer)
            target = unet.output_blocks[idx]
        else:
            assert False, "[DumpUnet] Invalid <Layer> value."
        
        features = defaultdict(list)
        def create_hook(features, name):
            def forward_hook(module, inputs, outputs):
                #print(f"{name}\t{inputs[0].size()}\t{outputs.size()}")
                self.steps_on_batch += 1
                if steps is None or self.steps_on_batch in steps:
                    list = features[self.steps_on_batch]
                    outputs = outputs.detach().clone()
                    for idx in range(outputs.size()[0] // 2): # two same outputs per sample???
                        output = outputs[idx]
                        list.append({
                            "name": name,
                            "input_dims": [ x.size() for x in inputs if type(x) == Tensor ],
                            "output_dims": output.size(),
                            "output": output,
                        })
            return forward_hook
        
        handles = []
        handles.append(target.register_forward_hook(create_hook(features, layer)))
        
        t0 = int(time.time())
        try:
            proc = process_images(p)
        finally:
            for handle in handles:
                handle.remove()
        
        if shared.state.interrupted:
            return proc
            
        index0 = proc.index_of_first_image
        preview_images, rest_images = proc.images[:index0], proc.images[index0:]
        
        assert rest_images is not None and len(rest_images) != 0, f"[DumpUnet] empty output?"
        
        # Now `rest_images` is the list of the images we are interested in.
        
        images = []
        seeds = []
        subseeds = []
        prompts = []
        neg_prompts = []
        infotexts = []
        
        def add_image(image, seed, subseed, prompt, neg_prompt, infotext, feature_steps=None):
            images.append(image)
            seeds.append(seed)
            subseeds.append(subseed)
            prompts.append(prompt)
            neg_prompts.append(neg_prompt)
            info = infotext
            if feature_steps is not None:
                if info:
                    info += "\n"
                info += f"Feature Steps: {feature_steps}"
            infotexts.append(info)
        
        for image in preview_images:
            preview_info = proc.infotexts.pop(0)
            add_image(image, proc.seed, proc.subseed, proc.prompt, proc.negative_prompt, preview_info)
        
        assert all([
            len(rest_images) == len(x) for x 
            in [proc.all_seeds, proc.all_subseeds, proc.all_prompts, proc.all_negative_prompts, proc.infotexts]
            ]), f"[DumpUnet] #images={len(rest_images)}, #seeds={len(proc.all_seeds)}, #subseeds={len(proc.all_subseeds)}, #pr={len(proc.all_prompts)}, #npr={len(proc.all_negative_prompts)}, #info={len(proc.infotexts)}"
        
        for idx, (image, *args) in enumerate(zip(rest_images, proc.all_seeds, proc.all_subseeds, proc.all_prompts, proc.all_negative_prompts, proc.infotexts)):
            add_image(image, *args)
            
            for step, fs in features.items():
                assert len(rest_images) == len(fs), f"[DumpUnet] #images={len(rest_images)}, #fs={len(fs)} @ index={idx}, step={step}."
                feature = fs[idx]
                
                if shared.state.interrupted:
                    break
            
                tensor = feature["output"]
                assert len(tensor.size()) == 3
                
                basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                canvases = process(tensor, grid_x, grid_y, tensor.size(), color, path, basename, path_on)
                
                for canvas in canvases:
                    add_image(canvas, *args, feature_steps=step)
        
        return Processed(
            p,
            images, 
            seed=proc.seed,
            info=proc.info,
            subseed=proc.subseed,
            all_seeds=seeds,
            all_subseeds=subseeds,
            all_prompts=prompts,
            all_negative_prompts=neg_prompts,
            infotexts=infotexts
        )

def process(tensor: Tensor,
            grid_x: int,
            grid_y: int,
            dims: tuple[int,int,int],
            color: bool,
            save_dir: str,
            basename: str,
            save_bin: bool = False
            ):
    # Regardless of wheather --opt-channelslast is enabled or not, 
    # feature.size() seems to return (batch, ch, h, w).
    # Is this intended result???
    
    max_ch, ih, iw = dims
    width = (grid_x * (iw + 1) - 1)
    height = (grid_y * (ih + 1) - 1)
    
    def each_slice(it: range, n: int):
        cur = []
        for x in it:
            cur.append(x)
            if n == len(cur):
                yield cur
                cur = []
        if 0 < len(cur):
            yield cur
    
    canvases = []
    color_format = "RGB" if color else "L"
    
    for chs in each_slice(range(max_ch), grid_x * grid_y):
        chs = list(chs)
        
        canvas = Image.new(color_format, (width, height), 0)
        for iy in range(grid_y):
            if len(chs) == 0:
                break
            
            for ix in range(grid_x):
                if shared.state.interrupted:
                    break
                
                if len(chs) == 0:
                    break
                
                ch = chs.pop(0)
                array = tensor[ch].cpu().numpy().astype(np.float32)
                filename = basename.format(x=ix, y=iy, ch=ch)
                
                # create image
                x = (iw+1) * ix
                y = (ih+1) * iy
                image = tensor_to_image(array, color)
                canvas.paste(Image.fromarray(image, color_format), (x, y))
                
                # save binary
                if save_bin:
                    assert save_dir is not None
                    binpath = os.path.join(save_dir, filename + ".bin")
                    with open(binpath, "wb") as io:
                        io.write(bytearray(array))
        
        canvases.append(canvas)
    return canvases

re_num = re.compile(r"^\s*\+?\s*\d+\s*$")
re_range = re.compile(r"^\s*(\+?\s*\d+)\s*-\s*(\+?\s*\d+)\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$")

def retrieve_steps(input: str):
    if input is None or input == "":
        return None
    
    steps : list[int]|None = []
    tokens = input.split(",")
    for token in tokens:
        if token == "":
            continue
        m1 = re_num.fullmatch(token)
        m2 = re_range.fullmatch(token)
        if m1:
            steps1 = eval("[" + m1.group(0) + "]")
        elif m2:
            n1 = eval(m2.group(1))
            n2 = eval(m2.group(2))
            n3 = eval(m2.group(3)) if m2.group(3) else 1
            steps1 = list(range(n1, n2+1, n3))
        else:
            raise ValueError("[DumpUnet] Invalid input for <Image saving steps>.")
        steps.extend(steps1)
    
    steps = list(set(steps))
    if len(steps) == 0:
        steps = None # all steps
    else:
        steps.sort()
    
    return steps

def get_grid_num(layer: str, width: int, height: int):
    assert layer is not None and layer != "", "[DumpUnet] <Layer> must not be empty."
    assert layer in LayerSettings, "[DumpUnet] Invalid <Layer> value."
    _, (ch, mh, mw) = LayerSettings[layer]
    iw = math.ceil(width  / 64)
    ih = math.ceil(height / 64)
    w = mw * iw
    h = mh * ih 
    # w : width of a feature map
    # h : height of a feature map
    # ch: a number of a feature map
    n = [w, h]
    while ch % 2 == 0:
        n[n[0]>n[1]] *= 2
        ch //= 2
    n[n[0]>n[1]] *= ch
    if n[0] > n[1]:
        while n[0] > n[1] * 2 and (n[0] // w) % 2 == 0:
            n[0] //= 2
            n[1] *= 2
    else:
        while n[0] * 2 < n[1] and (n[1] // h) % 2 == 0:
            n[0] *= 2
            n[1] //= 2
    
    return [n[0] // w, n[1] // h]

def tensor_to_image(array: np.ndarray, color: bool):
    # array := (-∞, ∞)
    
    if color:
        def colorize(v: float):
            # v = -1 .. 1 を
            # v < 0 のとき青 (0, 0, 1)
            # v > 0 のとき赤 (1 ,0, 0)
            # にする
            rgb = (v if v > 0.0 else 0.0, 0.0, -v if v < 0.0 else 0.0)
            return rgb
        colorize2 = np.vectorize(colorize, otypes=[np.float32, np.float32, np.float32])
        rgb = colorize2(np.clip(array, -1.0, 1.0))
        return np.clip((np.dstack(rgb) * 256), 0, 255).astype(np.uint8)
            
    else:
        return np.clip(np.abs(array) * 256, 0, 255).astype(np.uint8)
