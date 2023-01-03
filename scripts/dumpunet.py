import os
import time
import json
import re
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import TypeVar, Generator, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import process_images, Processed, StableDiffusionProcessing
from modules import shared

from scripts.dumpunet import layerinfo
from scripts.dumpunet.report import message as E

@dataclass
class FeatureInfo:
    input_dims: list[torch.Size]
    output_dims: torch.Size
    output: Tensor

class Features:
    
    # layer -> FeatureInfo
    features : dict[str, FeatureInfo]
    
    def __init__(self):
        self.features = dict()
    
    def __getitem__(self, layer: int|str):
        if isinstance(layer, int):
            return self.get_by_index(layer)
        elif isinstance(layer, str):
            return self.get_by_name
        else:
            raise ValueError(E(f"invalid type: {type(layer)}"))
    
    def __iter__(self):
        return sorted_items(self.features)
    
    def layers(self):
        return sorted_keys(self.features)
    
    def get_by_name(self, layer: str) -> FeatureInfo|None:
        if layer in self.features:
            return self.features[layer]
        return None
    
    def get_by_index(self, layer: int) -> FeatureInfo|None:
        name = layerinfo.name(layer)
        if name is None:
            return None
        return self.get_by_name(name)
    
    def add(self, layer: int|str, info: FeatureInfo):
        if isinstance(layer, int):
            name = layerinfo.name(layer)
            if name is None:
                raise ValueError(E(f"invalid layer name: {layer}"))
            layer = name
        self.features[layer] = info

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
                if steps is None or self.steps_on_batch in steps:
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

#re_layer_pat = r"((?:IN|OUT)(?:\d\d|@@|\$\$))"
re_layer_pat = r"((?:IN|OUT)\d\d|M00)"
re_layer = re.compile(rf"^\s*{re_layer_pat}\s*$")
re_layer_range = re.compile(rf"^\s*{re_layer_pat}\s*-\s*{re_layer_pat}\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$")

def retrieve_layers(input: str) -> list[str]:
    if input is None or input == "":
        # all layers
        return list(layerinfo.Names)
    
    def index(name: str):
        v = layerinfo.index(name)
        if v is None:
            raise ValueError(E(f"Invalid layer name: {name}"))
        return v
    
    result : list[int]|None = []
    tokens = input.split(",")
    for token in tokens:
        if token == "":
            continue
        m1 = re_layer.fullmatch(token)
        m2 = re_layer_range.fullmatch(token)
        if m1:
            result.append(index(m1.group(1)))
        elif m2:
            lay1 = index(m2.group(1))
            lay2 = index(m2.group(2))
            step = eval(m2.group(3)) if m2.group(3) else 1
            result.extend(range(lay1, lay2+1, step))
        else:
            raise ValueError(E(f"Invalid layer name: {token}"))
        
    result = list(set(result))
    if len(result) == 0:
        return list(layerinfo.Names)
    else:
        return [layerinfo.Names[n] for n in sorted(result)]

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
            raise ValueError(E(f"Invalid input for <Image saving steps>: {token}."))
        steps.extend(steps1)
    
    steps = list(set(steps))
    if len(steps) == 0:
        steps = None # all steps
    else:
        steps.sort()
    
    return steps

def get_grid_num(layer: str, width: int, height: int):
    assert layer is not None and layer != "", E("<Layers> must not be empty.")
    assert layer in layerinfo.Settings, E("Invalid <Layers> value.")
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

K = TypeVar('K')
V = TypeVar('V')

def sorted_items(obj: dict[K,V]|defaultdict[K,V]) -> Generator[Tuple[K,V],None,None]:
    for k in sorted_keys(obj):
        yield k, obj[k]

def sorted_keys(obj: dict[K,V]|defaultdict[K,V]) -> Generator[K,None,None]:
    for k in sorted(obj.keys()): # type: ignore
        yield k

def sorted_values(obj: dict[K,V]|defaultdict[K,V]) -> Generator[V,None,None]:
    for k in sorted_keys(obj):
        yield obj[k]
