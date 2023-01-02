import os
import time
import json
import re
import numpy as np
from torch import nn, Tensor
import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules.processing import process_images, Processed, StableDiffusionProcessing
from modules import shared

re_num = re.compile(r"^\s*\+?\s*\d+\s*$")
re_range = re.compile(r"^\s*(\+?\s*\d+)\s*-\s*(\+?\s*\d+)\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$")

class Script(scripts.Script):
    
    def title(self):
        return "Dump U-net features"
    
    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        settings = {
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
        
        with gr.Blocks(elem_id="dumpunet"):
            layer = gr.Dropdown([f"IN{i:02}" for i in range(12)] + ["M00"] + [f"OUT{i:02}" for i in range(12)], label="Layer", value="M00", elem_id="dumpunet-layer")
            layer_setting_hidden = gr.HTML(json.dumps(settings), visible=False, elem_id="dumpunet-layer_setting")
            
            with gr.Row():
                grid_x = gr.Slider(1, 512, value=1, step=1, label="Grid X", elem_id="dumpunet-gridx")
                grid_y = gr.Slider(1, 512, value=1, step=1, label="Grid Y", elem_id="dumpunet-gridy")
            
            steps = gr.Textbox(label="Image saving steps")
            
            color = gr.Checkbox(False, label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)") 
            
            with gr.Blocks():
                path_on = gr.Checkbox(False, label="Dump tensor to files")
                path = gr.Textbox(label="Output path")
            
            layer_info = gr.HTML(elem_id="dumpunet-layerinfo")
        
        return [layer, grid_x, grid_y, steps, color, path_on, path]
    
    def run(self,
            p: StableDiffusionProcessing,
            layer: str,
            grid_x: float,
            grid_y: float,
            step_input: str,
            color: bool,
            path_on: bool,
            path: str):
        
        IN = [ f"IN{i:02}" for i in range(12) ]
        OUT = [ f"OUT{i:02}" for i in range(12) ]
        
        assert p.n_iter == 1, "[DumpUnet] Batch count must be 1."
        assert p.batch_size == 1, "[DumpUnet] Batch size must be 1."
        assert layer is not None and layer != "", "[DumpUnet] <Layer> must not be empty."
        assert 1 <= grid_x, "[DumpUnet] <Grid X> must be positive integer."
        assert 1 <= grid_y, "[DumpUnet] <Grid Y> must be positive integer."
        if path_on:
            assert path is not None and path != "", "[DumpUnet] <Output path> must not be empty."
        
        steps : list[int]|None = []
        step_input_tokens = (step_input or "").split(",")
        for token in step_input_tokens:
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
        
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        
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
        
        features = []
        current_step = [0]
        def create_hook(features, name):
            def forward_hook(module, inputs, outputs):
                #print(f"{name}\t{inputs[0].size()}\t{outputs.size()}")
                current_step[0] += 1
                if steps is None or current_step[0] in steps:
                    features.append({
                        "steps": current_step[0],
                        "name": name,
                        "input_dims": [ x.size() for x in inputs if type(x) == Tensor ],
                        "output_dims": outputs.size(),
                        "outputs": outputs.detach().clone(),
                    })
            return forward_hook
        
        handles = []
        handles.append(target.register_forward_hook(create_hook(features, layer)))
        
        #for idx, mod in enumerate(input_blocks.children()):
        #    handles.append(mod.register_forward_hook(create_hook(features, f"IN{idx:02}")))
        #
        #handles.append(middle_block.register_forward_hook(create_hook(features, "M00")))
        #
        #for idx, mod in enumerate(output_blocks.children()):
        #    handles.append(mod.register_forward_hook(create_hook(features, f"OUT{idx:02}")))
        
        t0 = int(time.time())
        try:
            proc = process_images(p)
        finally:
            for handle in handles:
                handle.remove()
        
        if proc:
            assert len(proc.images) == 1, f"[DumpUnet] internal (#images={len(proc.images)}))"
            images = [proc.images[-1]]
            
            for step, feature in enumerate(features, 1):
                if shared.state.interrupted:
                    break
                
                tensors = feature["outputs"]
                assert len(tensors.size()) == 4
                for idx in range(tensors.size()[0]):
                    # two same outputs???
                    tensor = tensors[idx]
                    basename = f"{layer}-{step:03}-{{ch:04}}-{t0}"
                    canvases = process(tensor, grid_x, grid_y, tensor.size(), color, path, basename, path_on)
                    images.extend(canvases)
                    break
            
        else:
            images = proc.images
            
        N = lambda x: [x] * len(images)
        return Processed(
            p,
            images, 
            seed=proc.seed,
            info=proc.info,
            subseed=proc.subseed,
            all_seeds=N(proc.seed),
            all_subseeds=N(proc.subseed),
            all_prompts=N(proc.prompt),
            all_negative_prompts=N(proc.negative_prompt),
            infotexts=[proc.infotexts[0]] + [f"{proc.infotexts[0]}\nFeature Steps: {n}" for n in (steps or range(1, current_step[0]+1))]
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
