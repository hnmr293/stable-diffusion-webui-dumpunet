import os
import time
import json
import math
from collections import defaultdict

import numpy as np
from torch import nn, Tensor
import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import process_images, Processed, StableDiffusionProcessing
from modules import shared

from scripts.dumpunet import layerinfo
from scripts.dumpunet.features import Features, FeatureInfo
from scripts.dumpunet.ui import retrieve_layers, retrieve_steps
from scripts.dumpunet.report import message as E
from scripts.dumpunet.utils import *

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
            layer = gr.Textbox(label="Layers", placeholder="eg. IN00-OUT03(+2),OUT10", elem_id="dumpunet-layer")
            layer_setting_hidden = gr.HTML(json.dumps(layerinfo.Settings), visible=False, elem_id="dumpunet-layer_setting")
            steps = gr.Textbox(label="Image saving steps", placeholder="eg. 1,5-20(+5)")
            color = gr.Checkbox(False, label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)") 
            with gr.Blocks():
                path_on = gr.Checkbox(False, label="Dump tensor to files")
                path = gr.Textbox(label="Output path", placeholder="eg. /home/hnmr/unet/")
            layer_info = gr.HTML(elem_id="dumpunet-layerinfo")
        
        return [layer, steps, color, path_on, path]
    
    def process(self, p, *args):
        self.batch_num = 0
    
    def process_batch(self, p, *args, **kwargs):
        self.steps_on_batch = 0
        self.batch_num += 1
    
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
                 layer_input: str,
                 step_input: str,
                 color: bool,
                 path_on: bool,
                 path: str):
        
        assert layer_input is not None and layer_input != "", E("<Layers> must not be empty.")
        if path_on:
            assert path is not None and path != "", E("<Output path> must not be empty.")
        
        layers = retrieve_layers(layer_input)
        steps = retrieve_steps(step_input)
        if steps is None:
            steps = list(range(1, p.steps+1))
        
        unet = p.sd_model.model.diffusion_model # type: ignore
        
        #time_embed :  nn.modules.container.Sequential
        #input_blocks  : nn.modules.container.ModuleList
        #middle_block : ldm.modules.diffusionmodules.openaimodel.TimestepEmbedSequential
        #output_blocks : nn.modules.container.ModuleList
        #out_ : nn.modules.container.Sequential
        #time_embed = unet.time_embed
        #input_blocks = unet.input_blocks
        #middle_block = unet.middle_block
        #output_blocks = unet.output_blocks
        #out_ = unet.out
        #summary(unet, (4, 512, 512))
        
        # mkdir -p path
        if path_on:
            if os.path.exists(path):
                assert os.path.isdir(path), E("<Output path> already exists and is not a directory.")
            else:
                os.makedirs(path, exist_ok=True)
        
        # features : image_index -> step -> Features
        all_features : defaultdict[int, defaultdict[int, Features]]
        all_features = defaultdict(lambda: defaultdict(lambda: Features()))
        
        def start_step(module, inputs, outputs):
            self.steps_on_batch += 1
        
        def create_hook(all_features: defaultdict[int, defaultdict[int, Features]],
                        layername: str):
            def forward_hook(module, inputs, outputs):
                #print(f"{layername}\t{inputs[0].size()}\t{outputs.size()}")
                if self.steps_on_batch in steps:
                    images_per_batch = outputs.size()[0] // 2 # two same outputs per sample???
                    for image_index, output in enumerate(
                        outputs.detach().clone()[:images_per_batch],
                        (self.batch_num-1) * images_per_batch
                    ):
                        features = all_features[image_index][self.steps_on_batch]
                        features.add(layername, FeatureInfo(
                                [ x.size() for x in inputs if type(x) == Tensor ],
                                output.size(),
                                output))
            return forward_hook
        
        handles = []
        handles.append(unet.time_embed.register_forward_hook(start_step))
        for layer in layers:
            target = get_unet_layer(unet, layer)
            handles.append(target.register_forward_hook(create_hook(all_features, layer)))
        
        try:
            proc = process_images(p)
        finally:
            for handle in handles:
                handle.remove()
        
        if shared.state.interrupted:
            return proc
            
        index0 = proc.index_of_first_image
        preview_images, rest_images = proc.images[:index0], proc.images[index0:]
        
        assert rest_images is not None and len(rest_images) != 0, E("empty output?")
        
        # Now `rest_images` is the list of the images we are interested in.
        
        images = []
        seeds = []
        subseeds = []
        prompts = []
        neg_prompts = []
        infotexts = []
        
        def add_image(image, seed, subseed, prompt, neg_prompt, infotext, layername=None, feature_steps=None):
            images.append(image)
            seeds.append(seed)
            subseeds.append(subseed)
            prompts.append(prompt)
            neg_prompts.append(neg_prompt)
            info = infotext
            if layername is not None or feature_steps is not None:
                if info:
                    info += "\n"
                if layername is not None:
                    info += f"Layer Name: {layername}"
                if feature_steps is not None:
                    if layername is not None: info += ", "
                    info += f"Feature Steps: {feature_steps}"
                
            infotexts.append(info)
        
        for image in preview_images:
            preview_info = proc.infotexts.pop(0)
            add_image(image, proc.seed, proc.subseed, proc.prompt, proc.negative_prompt, preview_info)
        
        assert all([
            len(rest_images) == len(x) for x 
            in [proc.all_seeds, proc.all_subseeds, proc.all_prompts, proc.all_negative_prompts, proc.infotexts]
            ]), E(f"#images={len(rest_images)}, #seeds={len(proc.all_seeds)}, #subseeds={len(proc.all_subseeds)}, #pr={len(proc.all_prompts)}, #npr={len(proc.all_negative_prompts)}, #info={len(proc.infotexts)}")
        
        sorted_step_features = list(sorted_values(all_features))
        assert len(rest_images) == len(sorted_step_features), E(f"#images={len(rest_images)}, #features={len(sorted_step_features)}")
        
        t0 = int(time.time()) # for binary files' name
        shared.total_tqdm.clear()
        shared.total_tqdm.updateTotal(len(sorted_step_features) * len(steps) * len(layers))
        for idx, (image, step_features, *args) in enumerate(zip(rest_images, sorted_step_features, proc.all_seeds, proc.all_subseeds, proc.all_prompts, proc.all_negative_prompts, proc.infotexts)):
            step_features : defaultdict[int, Features]
            
            add_image(image, *args)
            
            for step, features in sorted_items(step_features):
                for layer, feature in features:
                    
                    if shared.state.interrupted:
                        break
                    
                    tensor = feature.output
                    assert len(tensor.size()) == 3
                    
                    grid_x, grid_y = get_grid_num(layer, p.width, p.height)
                    basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                    canvases = process(tensor, grid_x, grid_y, tensor.size(), color, path, basename, path_on)
                    
                    for canvas in canvases:
                        add_image(canvas, *args, layername=layer, feature_steps=step)
                    
                    shared.total_tqdm.update()
        
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
        
        canvas = Image.new(color_format, (width, height), color=0)
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

def get_unet_layer(unet, layername: str) -> nn.modules.Module:
    idx = layerinfo.input_index(layername)
    if idx is not None:
        return unet.input_blocks[idx]
    
    idx = layerinfo.middle_index(layername)
    if idx is not None:
        return unet.middle_block
    
    idx = layerinfo.output_index(layername)
    if idx is not None:
        return unet.output_blocks[idx]
    
    raise ValueError(E(f"Invalid layer name: {layername}"))

def get_grid_num(layer: str, width: int, height: int):
    assert layer is not None and layer != "", E("<Layers> must not be empty.")
    assert layer in layerinfo.Settings, E(f"Invalid <Layers> value: {layer}.")
    _, (ch, mh, mw) = layerinfo.Settings[layer]
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
