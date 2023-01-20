import sys
from typing import Any, Callable, TYPE_CHECKING

from torch import nn
from torch.utils.hooks import RemovableHandle

from modules.processing import StableDiffusionProcessing

from scripts.lib.features.featureinfo import MultiImageFeatures
from scripts.lib.report import message as E

if TYPE_CHECKING:
    from scripts.dumpunet import Script

class ExtractorBase:
    
    def __init__(self, runner: "Script", enabled: bool):
        self._runner = runner
        self._enabled = enabled
        self._handles: list[RemovableHandle] = []
        self._batch_num = 0
        self._steps_on_batch = 0
    
    @property
    def enabled(self):
        return self._enabled
    
    @property
    def batch_num(self):
        return self._batch_num
    
    @property
    def steps_on_batch(self):
        return self._steps_on_batch
    
    def __enter__(self):
        if self.enabled:
            self._batch_num = 0
            self._steps_on_batch = 0
            self._runner.on_process.add(self.on_process)
            self._runner.on_process_batch.add(self.on_process_batch)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            
            self._runner.on_process.remove(self.on_process)
            self._runner.on_process_batch.remove(self.on_process_batch)
            
            self.dispose()
    
    def dispose(self):
        pass
    
    def on_process(self, p: StableDiffusionProcessing, *args, **kwargs):
        self._batch_num = 0
        self._steps_on_batch = 0
    
    def on_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        self._batch_num += 1
        self._steps_on_batch = 0
    
    def on_process_step(self, module: nn.Module, inputs):
        self._steps_on_batch += 1
    
    def setup(
        self,
        p: StableDiffusionProcessing
    ):
        if not self.enabled:
            return
        
        wrapper = getattr(p.sd_model, "model", None)
        
        unet: nn.Module|None = getattr(wrapper, "diffusion_model", None) if wrapper is not None else None
        vae: nn.Module|None = getattr(p.sd_model, "first_stage_model", None)
        clip: nn.Module|None = getattr(p.sd_model, "cond_stage_model", None)
        
        assert unet is not None, E("p.sd_model.diffusion_model is not found. broken model???")
        self.hook_layer_pre(unet, self.on_process_step)
        
        self._do_hook(p, p.sd_model, unet=unet, vae=vae, clip=clip) # type: ignore
        self.on_setup()
    
    def on_setup(self):
        pass
    
    def _do_hook(
        self,
        p: StableDiffusionProcessing,
        model: Any,
        unet: nn.Module|None,
        vae: nn.Module|None,
        clip: nn.Module|None
    ):
        assert model is not None, E("empty model???")
        
        if clip is not None:
            self.hook_clip(p, clip)
        
        if unet is not None:
            self.hook_unet(p, unet)
        
        if vae is not None:
            self.hook_vae(p, vae)
    
    def hook_vae(
        self,
        p: StableDiffusionProcessing,
        vae: nn.Module
    ):
        pass

    def hook_unet(
        self,
        p: StableDiffusionProcessing,
        unet: nn.Module
    ):
        pass

    def hook_clip(
        self,
        p: StableDiffusionProcessing,
        clip: nn.Module
    ):
        pass

    def hook_layer(
        self,
        module: nn.Module|Any,
        fn: Callable[..., None]
    ):
        if not self.enabled:
            return
        
        assert module is not None
        assert isinstance(module, nn.Module)
        self._handles.append(module.register_forward_hook(fn))

    def hook_layer_pre(
        self,
        module: nn.Module|Any,
        fn: Callable[..., None]
    ):
        if not self.enabled:
            return
        
        assert module is not None
        assert isinstance(module, nn.Module)
        self._handles.append(module.register_forward_pre_hook(fn))

    def log(self, msg: str):
        if self._runner.debug:
            print(E(msg), file=sys.stderr)
