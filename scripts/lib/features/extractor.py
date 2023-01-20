import os, sys
import time
from typing import TYPE_CHECKING

from torch import nn, Tensor

from modules.processing import Processed, StableDiffusionProcessing
from modules import shared

from scripts.lib import layerinfo
from scripts.lib.extractor import FeatureExtractorBase
from scripts.lib.features.featureinfo import FeatureInfo, MultiImageFeatures
from scripts.lib.features.process import feature_to_grid_images, save_features
from scripts.lib.report import message as E
from scripts.lib.utils import *

if TYPE_CHECKING:
    from scripts.dumpunet import Script

class FeatureExtractor(FeatureExtractorBase[FeatureInfo]):
    
    def __init__(
        self,
        runner: "Script",
        enabled: bool,
        total_steps: int,
        layer_input: str,
        step_input: str,
        path: str|None,
    ):
        super().__init__(runner, enabled, total_steps, layer_input, step_input)
        
        if path is not None:
            assert path != "", E("<Output path> must not be empty.")
            # mkdir -p path
            if os.path.exists(path):
                assert os.path.isdir(path), E("<Output path> already exists and is not a directory.")
            else:
                os.makedirs(path, exist_ok=True)
        
        self.path = path
        
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        
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
        
        def create_hook(layername: str):
            
            def forward_hook(module, inputs, outputs):
                if self.steps_on_batch in self.steps:
                    self.log(f"{self.steps_on_batch} {layername} {inputs[0].size()} {outputs.size()}")
                    
                    images_per_batch = outputs.size()[0] // 2 # two same outputs per sample???
                    
                    for image_index, output in enumerate(
                        outputs.detach().clone()[:images_per_batch],
                        (self.batch_num-1) * images_per_batch
                    ):
                        features = self.extracted_features[image_index][self.steps_on_batch]
                        features.add(
                            layername,
                            FeatureInfo(
                                [ x.size() for x in inputs if type(x) == Tensor ],
                                output.size(),
                                output
                            )
                        )
            return forward_hook
        
        for layer in self.layers:
            self.log(f"U-Net: hooking {layer}...")
            target = get_unet_layer(unet, layer)
            self.hook_layer(target, create_hook(layer))
        
    def add_images(
        self,
        p: StableDiffusionProcessing,
        proc: Processed,
        extracted_features: MultiImageFeatures,
        color: bool
    ) -> Processed:
        
        if not self.enabled:
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
        
        # For Dynamic Prompt Extension
        # which is not append subseeds...
        while len(proc.all_subseeds) < len(proc.all_seeds):
            proc.all_subseeds.append(proc.all_subseeds[0] if 0 < len(proc.all_subseeds) else 0)
        
        assert all([
            len(rest_images) == len(x) for x 
            in [
                proc.all_seeds,
                proc.all_subseeds,
                proc.all_prompts,
                proc.all_negative_prompts,
                proc.infotexts
            ]
        ]), E(f"#images={len(rest_images)}, #seeds={len(proc.all_seeds)}, #subseeds={len(proc.all_subseeds)}, #pr={len(proc.all_prompts)}, #npr={len(proc.all_negative_prompts)}, #info={len(proc.infotexts)}")
        
        sorted_step_features = list(sorted_values(extracted_features))
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
                    
                    canvases = feature_to_grid_images(feature, layer, p.width, p.height, color)
                    for canvas in canvases:
                        add_image(canvas, *args, layername=layer, feature_steps=step)
                    
                    if self.path is not None:
                        basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                        save_features(feature, self.path, basename)
                    
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
