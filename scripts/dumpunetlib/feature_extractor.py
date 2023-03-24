import os
import time
from typing import TypeVar, Generic, TYPE_CHECKING

from modules import shared
from modules.processing import StableDiffusionProcessing, Processed

from scripts.dumpunetlib.extractor import ExtractorBase
from scripts.dumpunetlib.features.featureinfo import MultiImageFeatures
from scripts.dumpunetlib.ui import retrieve_layers, retrieve_steps
from scripts.dumpunetlib.putils import ProcessedBuilder
from scripts.dumpunetlib.report import message as E
from scripts.dumpunetlib.utils import sorted_values, sorted_items
from scripts.dumpunetlib.colorizer import Colorizer

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
        builder: ProcessedBuilder,
        extracted_features: MultiImageFeatures[TInfo],
        add_average: bool,
        color: Colorizer
    ):
        
        if not self.enabled:
            return
        
        if shared.state.interrupted:
            return
        
        sorted_step_features = list(sorted_values(extracted_features))
        assert len(builder.items) == len(sorted_step_features), E(f"#images={len(builder.items)}, #features={len(sorted_step_features)}")
        
        t0 = int(time.time()) # for binary files' name
        shared.total_tqdm.clear()
        shared.total_tqdm.updateTotal(len(sorted_step_features) * len(self.steps) * len(self.layers))
        
        for idx, step_features in enumerate(sorted_step_features):
            for step, features in sorted_items(step_features):
                for layer, feature in features:
                    
                    if shared.state.interrupted:
                        break
                    
                    canvases = self.feature_to_grid_images(feature, layer, idx, step, p.width, p.height, add_average, color)
                    for canvas in canvases:
                        builder.add_ref(idx, canvas, None, {"Layer Name": layer, "Feature Steps": step})
                    
                    if self.path is not None:
                        basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                        self.save_features(feature, layer, idx, step, p.width, p.height, self.path, basename)
                    
                    if hasattr(shared.total_tqdm, "_tqdm"):
                        shared.total_tqdm._tqdm.set_postfix_str(layer.ljust(5)) # type: ignore
                    shared.total_tqdm.update()
        
    def feature_to_grid_images(self, feature: TInfo, layer: str, img_idx: int, step: int, width: int, height: int, add_average: bool, color: Colorizer):
        raise NotImplementedError(f"{self.__class__.__name__}.feature_to_grid_images")
    
    def save_features(self, feature: TInfo, layer: str, img_idx: int, step: int, width: int, height: int, path: str, basename: str):
        raise NotImplementedError(f"{self.__class__.__name__}.save_features")
    
    def _fixup(self, proc: Processed):
        # For Dynamic Prompt Extension
        # which is not append subseeds...
        while len(proc.all_subseeds) < len(proc.all_seeds):
            proc.all_subseeds.append(proc.all_subseeds[0] if 0 < len(proc.all_subseeds) else 0)
        return proc
        
