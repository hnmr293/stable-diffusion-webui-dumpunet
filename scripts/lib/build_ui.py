from typing import Callable
from dataclasses import dataclass
from functools import partial
import json

from gradio import Blocks, Group, Tab, Row, Column, Textbox, Checkbox, HTML, Accordion

from scripts.lib import layerinfo

@dataclass
class OutputSetting:
    layers: Textbox
    steps: Textbox
    color: Checkbox
    
    @staticmethod
    def build(id: Callable[[str],str]):
        with Row():
            layers = Textbox(
                label="Layers",
                placeholder="eg. IN00-OUT03(+2),OUT10",
                elem_id=id("layer"),
            )
            
            steps = Textbox(
                label="Image saving steps",
                placeholder="eg. 1,5-20(+5)",
                elem_id=id("steps")
            )
        
        color = Checkbox(
            False,
            label="Use red/blue color map (red=POSITIVE, black=ZERO, blue=NEGATIVE)",
            elem_id=id("color")
        )
        color.style(container=False)
        
        return OutputSetting(
            layers,
            steps,
            color
        )

@dataclass
class DumpSetting:
    enabled: Checkbox
    path: Textbox
    
    @staticmethod
    def build(desc: str, id: Callable[[str],str]):
        enabled = Checkbox(
            False,
            label=desc,
            elem_id=id("dump-checkbox")
        )
        
        path = Textbox(
            label="Output path",
            placeholder="eg. /home/hnmr/unet/",
            elem_id=id("dumppath")
        )
        
        return DumpSetting(
            enabled,
            path
        )

@dataclass
class Info:
    selected: HTML
    all: HTML

@dataclass
class UNet:
    tab: Tab
    enabled: Checkbox
    settings: OutputSetting
    dump: DumpSetting
    info: Info

@dataclass
class LayerPrompt:
    tab: Tab
    enabled: Checkbox
    diff_enabled: Checkbox
    diff_settings: OutputSetting
    diff_dump: DumpSetting
    info: Info

@dataclass
class Debug:
    tab: Tab
    log: Checkbox

@dataclass
class UI:
    unet: UNet
    lp: LayerPrompt
    debug: Debug
    
    @staticmethod
    def build(runner, is_img2img: bool, id_prefix: str = "dumpunet"):
        
        def id(x: str, is_img2img: bool):
            return f"{id_prefix}-{['txt2img', 'img2img'][is_img2img]}-{x}"
        
        id = partial(id, is_img2img=is_img2img)
        
        with Group(elem_id=id("ui")):
            result = UI(
                build_unet(id),
                build_layerprompt(id),
                build_debug(runner, id),
            )
        
        return result
    
def build_unet(id_: Callable[[str],str]):
    id = lambda s: id_(f"features-{s}")
    
    with Tab("U-Net features", elem_id=id("tab")) as tab:
        enabled = Checkbox(
            label="Extract U-Net features",
            value=False,
            elem_id=id("checkbox")
        )
        
        settings = OutputSetting.build(id)
        
        with Accordion(label="Dump Setting", open=False):
            dump = DumpSetting.build("Dump feature tensors to files", id)
        
        info = build_info(id)
    
    return UNet(
        tab,
        enabled,
        settings,
        dump,
        info
    )

def build_layerprompt(id_: Callable[[str],str]):
    id = lambda s: id_(f"layerprompt-{s}")
    
    with Tab("Layer Prompt", elem_id=id("tab")) as tab:
        enabled = Checkbox(
            label="Enable Layer Prompt",
            value=False,
            elem_id=id("checkbox")
        )
        
        diff_enabled = Checkbox(
            label="Output difference map of U-Net features between with and without Layer Prompt",
            value=False,
            elem_id=id("diff-checkbox")
        )
            
        diff_settings = OutputSetting.build(lambda s: f"{id('diff')}-{s}")
        
        with Accordion(label="Dump Setting", open=False):
            diff_dump = DumpSetting.build("Dump difference tensors to files", lambda s: f"{id('diff')}-{s}")
        
        info = build_info(id)
            
    return LayerPrompt(
        tab,
        enabled,
        diff_enabled,
        diff_settings,
        diff_dump,
        info
    )

def build_debug(runner, id: Callable[[str],str]):
    with Tab("Settings") as tab:
        debug = Checkbox(
            label="log to stderr",
            value=runner.debug
        )
        
        def set_debug(x):
            runner.debug = x

        debug.change(
            fn=set_debug,
            inputs=debug
        )
    
    return Debug(
        tab,
        debug
    )

def build_info(id: Callable[[str],str]):
    with Accordion("Selected Layer Info", open=False):
        info = HTML(elem_id=id("layerinfo"))
        all_settings_hidden = HTML(
            json.dumps(layerinfo.Settings),
            visible=False,
            elem_id=id("layer_setting")
        )
    
    return Info(info, all_settings_hidden)
