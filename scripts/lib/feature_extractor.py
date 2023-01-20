import os
import time
from typing import TypeVar, Generic, TYPE_CHECKING

from modules import shared
from modules.processing import StableDiffusionProcessing, Processed

from scripts.lib.extractor import ExtractorBase
from scripts.lib.features.featureinfo import MultiImageFeatures
from scripts.lib.ui import retrieve_layers, retrieve_steps
from scripts.lib.putils import ProcessedBuilder
from scripts.lib.report import message as E
from scripts.lib.utils import sorted_values, sorted_items

if TYPE_CHECKING:
    from scripts.dumpunet import Script

TInfo = TypeVar("TInfo")

class FeatureExtractorBase(Generic[TInfo], ExtractorBase):
    
    # image_index -> step -> Features
    extracted_features: MultiImageFeatures[TInfo]
    
    # steps to process
    steps: list[int]
    
    # layers to process
    layers: list[str]
    
    # dump path
    path: str|None
    
    def __init__(
        self,
        runner: "Script",
        enabled: bool,
        total_steps: int,
        layer_input: str,
        step_input: str,
        path: str|None
    ):
        super().__init__(runner, enabled)
        
        self.extracted_features = MultiImageFeatures()
        self.steps = []
        self.layers = []
        self.path = None
        
        if not self.enabled:
            return
        
        assert layer_input is not None and layer_input != "", E("<Layers> must not be empty.")
        
        self.layers = retrieve_layers(layer_input)
        self.steps = (
            retrieve_steps(step_input) 
            or list(range(1, total_steps+1))
        )
        
        if path is not None:
            assert path != "", E("<Output path> must not be empty.")
            # mkdir -p path
            if os.path.exists(path):
                assert os.path.isdir(path), E("<Output path> already exists and is not a directory.")
            else:
                os.makedirs(path, exist_ok=True)
        
        self.path = path
    
    def on_setup(self):
        self.extracted_features = MultiImageFeatures()
    
    def add_images(
        self,
        p: StableDiffusionProcessing,
        proc: Processed,
        extracted_features: MultiImageFeatures[TInfo],
        color: bool
    ):
        
        if not self.enabled:
            return proc
        
        if shared.state.interrupted:
            return proc
        
        self._fixup(proc)
        index0 = proc.index_of_first_image
        preview_images, rest_images = proc.images[:index0], proc.images[index0:]
        
        assert rest_images is not None and len(rest_images) != 0, E("empty output?")
        
        # Now `rest_images` is the list of the images we are interested in.
        
        builder = ProcessedBuilder()
        for image in preview_images:
            preview_info = proc.infotexts.pop(0)
            builder.add(image, proc.seed, proc.subseed, proc.prompt, proc.negative_prompt, preview_info)
        
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
            builder.add(image, *args)
            
            for step, features in sorted_items(step_features):
                for layer, feature in features:
                    
                    if shared.state.interrupted:
                        break
                    
                    canvases = self.feature_to_grid_images(feature, layer, p.width, p.height, color)
                    for canvas in canvases:
                        builder.add(canvas, *args, {"Layer Name": layer, "Feature Steps": step})
                    
                    if self.path is not None:
                        basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                        self.save_features(feature, self.path, basename)
                    
                    shared.total_tqdm.update()
        
        return builder.to_proc(p, proc)
        
    def feature_to_grid_images(self, feature: TInfo, layer: str, width: int, height: int, color: bool):
        raise NotImplementedError(f"{self.__class__.__name__}.feature_to_grid_images")
    
    def save_features(self, feature: TInfo, path: str, basename: str):
        raise NotImplementedError(f"{self.__class__.__name__}.save_features")
    
    def _fixup(self, proc: Processed):
        # For Dynamic Prompt Extension
        # which is not append subseeds...
        while len(proc.all_subseeds) < len(proc.all_seeds):
            proc.all_subseeds.append(proc.all_subseeds[0] if 0 < len(proc.all_subseeds) else 0)
        return proc
        
