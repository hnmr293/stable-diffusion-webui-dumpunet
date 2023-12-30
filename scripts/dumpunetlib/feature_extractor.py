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

try:
    import modules.images
    module_images_loaded = True
except ImportError:
    module_images_loaded = False

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

    # image saving path
    image_path: str|None
    
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

        if image_path is not None:
            assert image_path != "", E("<Image path> must not be empty.")
            # mkdir -p image_path
            if os.path.exists(image_path):
                assert os.path.isdir(image_path), E("<Image path> already exists and is not a directory.")
            else:
                os.makedirs(image_path, exist_ok=True)
        
        self.image_path = image_path
    
    def on_setup(self):
        self.extracted_features = MultiImageFeatures()
    
    def add_images(
        self,
        p: StableDiffusionProcessing,
        builder: ProcessedBuilder,
        extracted_features: MultiImageFeatures[TInfo],
        average_type: str|None,
        color: Colorizer,
        name: str = "",
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
                    
                    canvases = self.feature_to_grid_images(feature, layer, idx, step, p.width, p.height, average_type, color)
                    for canvas in canvases:
                        builder.add_ref(idx, canvas, None, {"Layer Name": layer, "Feature Steps": step})
                        self._save_generated_image(p, canvas, name, idx, layer, step)
                    
                    if self.path is not None:
                        basename = f"{idx:03}-{layer}-{step:03}-{{ch:04}}-{t0}"
                        self.save_features(feature, layer, idx, step, p.width, p.height, self.path, basename)
                    
                    if hasattr(shared.total_tqdm, "_tqdm"):
                        shared.total_tqdm._tqdm.set_postfix_str(layer.ljust(5)) # type: ignore
                    shared.total_tqdm.update()
        
    def feature_to_grid_images(self, feature: TInfo, layer: str, img_idx: int, step: int, width: int, height: int, average_type: str|None, color: Colorizer):
        raise NotImplementedError(f"{self.__class__.__name__}.feature_to_grid_images")
    
    def save_features(self, feature: TInfo, layer: str, img_idx: int, step: int, width: int, height: int, path: str, basename: str):
        raise NotImplementedError(f"{self.__class__.__name__}.save_features")
    
    def _fixup(self, proc: Processed):
        # For Dynamic Prompt Extension
        # which is not append subseeds...
        while len(proc.all_subseeds) < len(proc.all_seeds):
            proc.all_subseeds.append(proc.all_subseeds[0] if 0 < len(proc.all_subseeds) else 0)
        return proc
    
    def _save_generated_image(self, p, image, prefix: str, image_index: int, layer: str, step: int):
        if module_images_loaded and self.image_path:
            if prefix is None or len(prefix) == 0:
                basename = f"-dumpunet-{layer}-{step:03}"
            else:
                basename = f"-dumpunet-{prefix}-{layer}-{step:03}"
            
            orig = modules.images.get_next_sequence_number
            try:
                # hook image number
                #def get_next_sequence_number(path: str, basename: str):
                #    # in processing.py, `images.save_image` is called with one of
                #    #   path = p.outpath_samples
                #    #          p.outpath_grids (for grid)
                #    #          opts.outdir_init_images (for img2img)
                #    # so the target image of a image saving here will be stored
                #    # always in p.outpath_samples.
                #    basecount = orig(p.outpath_samples, basename)
                #    return basecount - 1
                assert self.image_path == p.outpath_samples, E(f"not implemented (image_path={repr(self.image_path)})")
                def get_next_sequence_number(*args, **kwargs):
                    basecount = orig(*args, **kwargs)
                    return basecount - 1
                modules.images.get_next_sequence_number = get_next_sequence_number
                
                modules.images.save_image(
                    image,
                    self.image_path,
                    "",
                    p.seeds[image_index],
                    p.prompts[image_index],
                    shared.opts.samples_format,
                    p=p,
                    suffix=basename
                )

            finally:
                modules.images.get_next_sequence_number = orig
