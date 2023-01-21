from typing import Callable
from dataclasses import dataclass
from functools import partial
import json

from gradio import Blocks, Group, Tab, Row, Column, Textbox, Checkbox, Radio, Slider, Number, HTML, Accordion

from scripts.lib import layerinfo

@dataclass
class OutputSetting:
    layers: Textbox
    steps: Textbox
    colorize: Radio
    colorspace: Radio
    R: Textbox
    G: Textbox
    B: Textbox
    H: Textbox
    S: Textbox
    L: Textbox
    trans: Radio
    linear_min: Slider
    linear_max: Slider
    sigmoid_gain: Slider
    sigmoid_offset: Slider
    
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
        
        with Accordion("Colorization"):
            show = {"visible": True,  "__type__": "update"}
            hide = {"visible": False, "__type__": "update"}
            
            colorize = Radio(choices=["White/Black", "Red/Blue", "Custom"], value="White/Black", label="Colorize method")
            trans = Radio(choices=["Linear", "Sigmoid"], value="Sigmoid", label="Value transform")
            
            with Row(visible=False) as linear_option:
                clamp_min = Slider(minimum=-10, maximum=-0.1, value=-1, step=0.1, label="Clamp min.", interactive=True)
                clamp_max = Slider(minimum=0.1, maximum=10, value=1, step=0.1, label="Clamp max.", interactive=True)
            
            with Row(visible=True) as sigmoid_option:
                sigmoid_gain = Slider(minimum=0.1, maximum=2, value=1.0, step=0.1, label="gain", interactive=True)
                sigmoid_offset = Slider(minimum=-10, maximum=10, value=0.0, step=0.1, label="offset X", interactive=True)
            map(lambda x: x.style(container=False), [clamp_min, clamp_max, sigmoid_gain, sigmoid_offset])
            
            with Group(visible=False) as colorize_custom_option:
                colorspace = Radio(choices=["RGB", "HSL"], value="RGB", label="Color space")
                with Row(visible=True) as RGB:
                    r = Textbox(value="abs(v)", label="R", interactive=True)
                    g = Textbox(value="abs(v)", label="G", interactive=True)
                    b = Textbox(value="abs(v)", label="B", interactive=True)
                with Row(visible=False) as HSL:
                    h = Textbox(value="(2+v)/3", label="H", interactive=True)
                    s = Textbox(value="1.0", label="S", interactive=True)
                    l = Textbox(value="0.5", label="L", interactive=True)
                map(lambda x: x.style(container=False), [r,g,b,h,s,l])
            
            def color_change(x):
                if x == "Custom":
                    return show
                else:
                    return hide
            def trans_change(x):
                if x == "Linear":
                    return show, hide
                else:
                    return hide, show
            def colorspace_change(x):
                if x == "RGB":
                    return show, hide
                else:
                    return hide, show
            colorize.change(fn=color_change, inputs=[colorize], outputs=[colorize_custom_option], show_progress=False) # type: ignore
            trans.change(fn=trans_change, inputs=[trans], outputs=[linear_option, sigmoid_option], show_progress=False) # type: ignore
            colorspace.change(fn=colorspace_change, inputs=[colorspace], outputs=[RGB, HSL], show_progress=False) # type: ignore
            
        return OutputSetting(
            layers,
            steps,
            colorize,
            colorspace,
            r, g, b,
            h, s, l,
            trans,
            clamp_min, clamp_max,
            sigmoid_gain, sigmoid_offset
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
class Attn:
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
    attn: Attn
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
                build_attn(id),
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

def build_attn(id_: Callable[[str],str]):
    id = lambda s: id_(f"attention-{s}")
    
    with Tab("Attention", elem_id=id("tab")) as tab:
        enabled = Checkbox(
            label="Extract attention layers' features",
            value=False,
            elem_id=id("checkbox")
        )
        
        settings = OutputSetting.build(id)
        
        with Accordion(label="Dump Setting", open=False):
            dump = DumpSetting.build("Dump feature tensors to files", id)
        
        info = build_info(id)
    
    return Attn(
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
