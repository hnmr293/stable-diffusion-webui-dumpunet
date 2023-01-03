import os
import time
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable

import torch
from torch import nn, Tensor
from torch.utils.hooks import RemovableHandle
import numpy as np
from PIL import Image

from modules.processing import Processed, StableDiffusionProcessing
from modules import shared

from scripts.dumpunet import layerinfo
from scripts.dumpunet.ui import retrieve_layers, retrieve_steps
from scripts.dumpunet.report import message as E
from scripts.dumpunet.utils import *

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
            return self.get_by_name(layer)
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

class FeatureExtractor:
    
    # image_index -> step -> Features
    extracted_features: defaultdict[int, defaultdict[int, Features]]
    
    # steps to process
    steps: list[int]
    
    # layers to process
    layers: list[str]
    
    def __init__(
        self,
        runner,
        enabled: bool,
        total_steps: int,
        layer_input: str,
        step_input: str,
        path: str|None,
    ):
        self._runner = runner
        self._enabled = enabled
        self._handles: list[RemovableHandle] = []
        
        self.extracted_features = defaultdict(lambda: defaultdict(lambda: Features()))
        self.steps = []
        self.layers = []
        self.path = None
        
        if not self._enabled:
            return
        
        assert layer_input is not None and layer_input != "", E("<Layers> must not be empty.")
        if path is not None:
            assert path != "", E("<Output path> must not be empty.")
            # mkdir -p path
            if os.path.exists(path):
                assert os.path.isdir(path), E("<Output path> already exists and is not a directory.")
            else:
                os.makedirs(path, exist_ok=True)
        
        self.layers = retrieve_layers(layer_input)
        self.steps = (
            retrieve_steps(step_input) 
            or list(range(1, total_steps+1))
        )
        self.path = path
        
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def setup(
        self,
        p: StableDiffusionProcessing,
    ):
        if not self._enabled:
            return
        
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
        
        def start_step(module, inputs, outputs):
            self._runner.steps_on_batch += 1
        
        def create_hook(layername: str):
            
            def forward_hook(module, inputs, outputs):
                #print(f"{layername}\t{inputs[0].size()}\t{outputs.size()}")
                
                if self._runner.steps_on_batch in self.steps:
                    images_per_batch = outputs.size()[0] // 2 # two same outputs per sample???
                    
                    for image_index, output in enumerate(
                        outputs.detach().clone()[:images_per_batch],
                        (self._runner.batch_num-1) * images_per_batch
                    ):
                        features = self.extracted_features[image_index][self._runner.steps_on_batch]
                        features.add(
                            layername,
                            FeatureInfo(
                                [ x.size() for x in inputs if type(x) == Tensor ],
                                output.size(),
                                output
                            )
                        )
            return forward_hook
        
        self._handles.append(unet.time_embed.register_forward_hook(start_step))
        for layer in self.layers:
            target = get_unet_layer(unet, layer)
            self._handles.append(target.register_forward_hook(create_hook(layer)))
        
    def create_feature_map(
        self,
        p: StableDiffusionProcessing,
        proc: Processed,
        color: bool
    ):
        if not self._enabled:
            return proc
        
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
        
        sorted_step_features = list(sorted_values(self.extracted_features))
        assert len(rest_images) == len(sorted_step_features), E(f"#images={len(rest_images)}, #features={len(sorted_step_features)}")
        
        t0 = int(time.time()) # for binary files' name
        shared.total_tqdm.clear()
        shared.total_tqdm.updateTotal(len(sorted_step_features) * len(self.steps) * len(self.layers))
        
        image_args = zip(
            proc.all_seeds,
            proc.all_subseeds,
            proc.all_prompts,
            proc.all_negative_prompts,
            proc.infotexts
        )
        
        for idx, (image, step_features, args) in enumerate(zip(rest_images, sorted_step_features, image_args)):
            add_image(image, *args)
            
            for step, features in sorted_items(step_features):
                for layer, feature in features:
                    
                    if shared.state.interrupted:
                        break
                    
                    tensor = feature.output
                    assert len(tensor.size()) == 3
                    
                    grid_x, grid_y = get_grid_num(layer, p.width, p.height)
                    basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                    canvases = process(tensor, grid_x, grid_y, tensor.size(), color, self.path or "", basename, self.path is not None)
                    
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
            infotexts=infotexts,
        )

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
