import sys
import math
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor, einsum
from einops import rearrange

from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention, MemoryEfficientCrossAttention # type: ignore
from modules.processing import StableDiffusionProcessing
from modules.hypernetworks import hypernetwork
from modules import shared

from scripts.dumpunetlib.feature_extractor import FeatureExtractorBase
from scripts.dumpunetlib.features.featureinfo import MultiImageFeatures
from scripts.dumpunetlib.features.extractor import get_unet_layer
from scripts.dumpunetlib.attention.featureinfo import AttnFeatureInfo
from scripts.dumpunetlib import layerinfo, tutils
from scripts.dumpunetlib.colorizer import Colorizer
from scripts.dumpunetlib.utils import *
from scripts.dumpunetlib.report import message as E

if TYPE_CHECKING:
    from scripts.dumpunet import Script

class AttentionExtractor(FeatureExtractorBase):

    features_to_save: list[str]
    
    # image_index -> step -> Features
    extracted_features: MultiImageFeatures[AttnFeatureInfo]
    
    def __init__(
        self,
        runner: "Script",
        enabled: bool,
        total_steps: int,
        layer_input: str,
        step_input: str,
        features: list[str],
        path: str|None,
    ):
        if features is None or len(features) == 0:
            if enabled:
                enabled = False
                print("\033[33m", file=sys.stderr, end="", flush=False)
                print(E("Attention: Disabled because no features are selected. Select features in <Output features>."), file=sys.stderr, end="", flush=False)
                print("\033[0m", file=sys.stderr)
        
        super().__init__(runner, enabled, total_steps, layer_input, step_input, path)
        self.features_to_save = features
        self.extracted_features = MultiImageFeatures()
    
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        
        def create_hook(layername: str, block: BasicTransformerBlock,  n: int, depth: int, c: int):
            
            def forward(module, fn, x, context=None, *args, **kwargs):
                result = fn(x, context=context, *args, **kwargs)
                
                if self.steps_on_batch in self.steps:
                    if c == 2:
                        # process for only cross-attention
                        self.log(f"{self.steps_on_batch:>03} {layername}-{n}-{depth}-attn{c} ({'cross' if (block.disable_self_attn or 1 < c) else 'self'})")
                        self.log(f"    | {shape(x),shape(context)} -> {shape(result)}")
                        
                        ks, qks, vqks = self.process_attention(module, x, context)
                        # qk := (batch, head, token, height*width)
                        # vqk := (batch, height*width, ch)
                        
                        images_per_batch = p.batch_size
                        assert qks.shape[0] == vqks.shape[0], f"{qks.shape}, {vqks.shape}"
                        
                        for image_index, (k, vk, vqk) in enumerate(
                            zip(ks[:images_per_batch], qks[:images_per_batch], vqks[:images_per_batch]),
                            (self.batch_num-1) * images_per_batch
                        ):
                            features = self.extracted_features[image_index][self.steps_on_batch]
                            features.add(
                                layername,
                                AttnFeatureInfo(k, vk, vqk)
                            )
                
                return result
            
            return forward
        
        active_steps: list[str] = []
        for layer in self.layers:
            for n, d, block, attn1, attn2 in get_unet_attn_layers(unet, layer):
                self.log(f"Attention: hooking {layer}...")
                active_steps.append(layer)
                self.hook_forward(attn1, create_hook(layer, block, n, d, 1))
                self.hook_forward(attn2, create_hook(layer, block, n, d, 2))
        self.layers = sorted(set(active_steps), key=active_steps.index)
        
        return super().hook_unet(p, unet)
    
    def process_attention(self, module, x, context):
        # q_in : unet features ([2, 4096, 320])
        # k_in, v_in : embedding vector kv (cross-attention) ([2, 77, 320]) or unet features kv (self-attention) ([2, 4096, 320])
        # q,k,v : head-separated q_in, k_in and v_in
        
        if getattr(hypernetwork, "apply_hypernetworks", None) is not None:
            ctx_k, ctx_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
        elif getattr(hypernetwork, "apply_hypernetwork", None) is not None:
            ctx_k, ctx_v = hypernetwork.apply_hypernetwork( # type: ignore
                shared.loaded_hypernetwork, # type: ignore
                context if context is not None else x
            )
        else:
            assert False, "not supported version"
        
        q_in = module.to_q(x)
        k_in = module.to_k(ctx_k)
        v_in = module.to_v(ctx_v)
        
        q: Tensor
        k: Tensor
        v: Tensor
        
        q, k, v = map( # type: ignore
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=module.heads),
            (q_in, k_in, v_in)
        )
        
        sim = einsum('b i d, b j d -> b i j', q, k) * module.scale
        sim = sim.softmax(dim=-1)
        # sim.shape == '(b h) i j'
        
        o_in = einsum('b i j, b j d -> b i d', sim, v)
        o = rearrange(o_in, '(b h) n d -> b n (h d)', h=module.heads)
        
        kk: Tensor = rearrange(k, '(b h) t d -> b h t d', h=module.heads).detach().clone()
        qk: Tensor = rearrange(sim, '(b h) d t -> b h t d', h=module.heads).detach().clone()
        vqk: Tensor = o.detach().clone()
        
        self.log(f"    | q: {shape(q_in)} # {shape(q)}")
        self.log(f"    | k: {shape(k_in)} # {shape(k)}")
        self.log(f"    | v: {shape(v_in)} # {shape(v)}")
        #self.log(f"    | kk: {shape(kk)} # {shape(k)}")
        #self.log(f"    | qk: {shape(qk)} # {shape(sim)}")
        #self.log(f"    | vqk: {shape(vqk)}")
        
        del q_in, k_in, v_in, q, k, v, sim, o_in, o
        
        return kk, qk, vqk
    
    def add_images(
        self,
        p: StableDiffusionProcessing,
        builder,
        extracted_features: MultiImageFeatures[AttnFeatureInfo],
        average_type: str|None,
        color: Colorizer
    ):
        if not self.enabled: return
        if shared.state.interrupted: return
        if len(extracted_features) == 0:
            print("\033[33m", file=sys.stderr, end="", flush=False)
            print(E("Attention: no images are extracted"), file=sys.stderr, end="", flush=False)
            print("\033[0m", file=sys.stderr)
            return
        return super().add_images(p, builder, extracted_features, average_type, color)
    
    def feature_to_grid_images(self, feature: AttnFeatureInfo, layer: str, img_idx: int, step: int, width: int, height: int, average_type: str|None, color: Colorizer):
        w, h, ch = get_shape(layer, width, height)
        images = []
        
        if "K" in self.features_to_save:
            k = feature.k
            heads_k, ch_k, n_k = k.shape
            assert ch_k % 77 == 0, f"ch_k={ch_k}"
            k1 = rearrange(k, 'a t n -> a n t').contiguous()
            k_images = tutils.tensor_to_image(k1, 1, heads_k, color, average_type)
            images.extend(k_images)
            del k1
        
        if "Q*K" in self.features_to_save:
            qk = feature.qk
            heads_qk, ch_qk, n_qk = qk.shape
            assert ch_qk % 77 == 0, f"ch_qk={ch_qk}"
            assert w * h == n_qk, f"w={w}, h={h}, n_qk={n_qk}"
            qk1 = rearrange(qk, 'a t (h w) -> (a t) h w', h=h).contiguous()
            qk_images = tutils.tensor_to_image(qk1, ch_qk, heads_qk, color, average_type=None)
            qk_avg = tutils.averaged_tensor(
                rearrange(qk, 'a t (h w) -> a t h w', h=h).contiguous(),
                average_type
            )
            if qk_avg is not None:
                qk_avg_image = tutils.tensor_to_image(qk_avg, ch_qk, 1, color, average_type=None)
                qk_images = qk_avg_image + qk_images
            images.extend(qk_images)
            del qk1
        
        if "V*Q*K" in self.features_to_save:
            vqk = feature.vqk
            n_vqk, ch_vqk = vqk.shape
            assert w * h == n_vqk, f"w={w}, h={h}, n_vqk={n_vqk}"
            assert ch == ch_vqk, f"ch={ch}, ch_vqk={ch_vqk}"
            vqk1 = rearrange(vqk, '(h w) c -> c h w', h=h).contiguous()
            vqk_images = tutils.tensor_to_grid_images(vqk1, layer, width, height, color, average_type)
            images.extend(vqk_images)
            del vqk1
        
        return images
    
    def save_features(self, feature: AttnFeatureInfo, layer: str, img_idx: int, step: int, width: int, height: int, path: str, basename: str):
        w, h, ch = get_shape(layer, width, height)
        qk = rearrange(feature.qk, 'a t (h w) -> (a t) h w', h=h).contiguous()
        tutils.save_tensor(qk, path, basename)
    
def get_shape(layer: str, width: int, height: int):
    assert layer in layerinfo.Settings
    (ich, ih, iw), (och, oh, ow) = layerinfo.Settings[layer]
    nw, nh = [max(1, math.ceil(x / 64)) for x in [width, height]]
    return iw*nw, ih*nh, och

def get_unet_attn_layers(unet, layername: str):
    unet_block = get_unet_layer(unet, layername)
    
    def each_transformer(unet_block):
        for block in unet_block.children():
            if isinstance(block, SpatialTransformer):
                yield block
    
    def each_basic_block(trans):
        for block in trans.children():
            if isinstance(block, BasicTransformerBlock):
                yield block
    
    for n, trans in enumerate(each_transformer(unet_block)):
        for depth, basic_block in enumerate(each_basic_block(trans.transformer_blocks)):
            attn1: CrossAttention|MemoryEfficientCrossAttention
            attn2: CrossAttention|MemoryEfficientCrossAttention
            
            attn1, attn2 = basic_block.attn1, basic_block.attn2
            assert isinstance(attn1, CrossAttention) or isinstance(attn1, MemoryEfficientCrossAttention)
            assert isinstance(attn2, CrossAttention) or isinstance(attn2, MemoryEfficientCrossAttention)
            
            yield n, depth, basic_block, attn1, attn2

def shape(t: Tensor|None) -> tuple|None:
    return tuple(t.shape) if t is not None else None
