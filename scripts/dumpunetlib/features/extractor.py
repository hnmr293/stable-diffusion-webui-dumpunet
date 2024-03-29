from typing import TYPE_CHECKING

from torch import nn, Tensor

from modules.processing import StableDiffusionProcessing

from scripts.dumpunetlib import layerinfo
from scripts.dumpunetlib.feature_extractor import FeatureExtractorBase
from scripts.dumpunetlib.features.featureinfo import FeatureInfo
from scripts.dumpunetlib.features.utils import feature_to_grid_images, save_features
from scripts.dumpunetlib.report import message as E
from scripts.dumpunetlib.utils import *
from scripts.dumpunetlib.colorizer import Colorizer

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
        image_path: str|None,
    ):
        super().__init__(runner, enabled, total_steps, layer_input, step_input, path, image_path)
        
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
                    
                    images_per_batch = p.batch_size
                    
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
    
    def feature_to_grid_images(self, feature: FeatureInfo, layer: str, img_idx: int, step: int, width: int, height: int, average_type: str|None, color: Colorizer):
        return feature_to_grid_images(feature, layer, width, height, average_type, color)
    
    def save_features(self, feature: FeatureInfo, layer: str, img_idx: int, step: int, width: int, height: int, path: str, basename: str):
        save_features(feature, path, basename)
    
    def add_images(self, *args, **kwargs):
        kwargs['name'] = 'unet'
        super().add_images(*args, **kwargs)
        
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
