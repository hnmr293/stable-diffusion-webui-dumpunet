import math
from typing import TYPE_CHECKING

from torch import nn, Tensor, einsum
from einops import rearrange

from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention, MemoryEfficientCrossAttention # type: ignore
from modules.processing import StableDiffusionProcessing
from modules.hypernetworks import hypernetwork
from modules import shared

from scripts.lib.feature_extractor import FeatureExtractorBase
from scripts.lib.features.featureinfo import MultiImageFeatures
from scripts.lib.features.extractor import get_unet_layer
from scripts.lib.attention.featureinfo import AttnFeatureInfo
from scripts.lib import layerinfo, tutils
from scripts.lib.colorizer import Colorizer
from scripts.lib.utils import *

if TYPE_CHECKING:
    from scripts.dumpunet import Script

class AttentionExtractor(FeatureExtractorBase):

    # image_index -> step -> Features
    extracted_features: MultiImageFeatures[AttnFeatureInfo]
    
    def __init__(
        self,
        runner: "Script",
        enabled: bool,
        total_steps: int,
        layer_input: str,
        step_input: str,
        path: str|None,
    ):
        super().__init__(runner, enabled, total_steps, layer_input, step_input, path)
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
                        
                        qks, vqks = self.process_attention(module, x, context)
                        # qk := (batch, head, token, height*width)
                        # vqk := (batch, height*width, ch)
                        
                        images_per_batch = qks.shape[0] // 2
                        assert qks.shape[0] == vqks.shape[0]
                        assert qks.shape[0] % 2 == 0
                        
                        for image_index, (vk, vqk) in enumerate(
                            zip(qks[:images_per_batch], vqks[:images_per_batch]),
                            (self.batch_num-1) * images_per_batch
                        ):
                            features = self.extracted_features[image_index][self.steps_on_batch]
                            features.add(
                                layername,
                                AttnFeatureInfo(vk, vqk)
                            )
                
                return result
            
            return forward
        
        for layer in self.layers:
            self.log(f"Attention: hooking {layer}...")
            for n, d, block, attn1, attn2 in get_unet_attn_layers(unet, layer):
                self.hook_forward(attn1, create_hook(layer, block, n, d, 1))
                self.hook_forward(attn2, create_hook(layer, block, n, d, 2))
        
        return super().hook_unet(p, unet)
    
    def process_attention(self, module, x, context):
        # q_in : unet features ([2, 4096, 320])
        # k_in, v_in : embedding vector kv (cross-attention) ([2, 77, 320]) or unet features kv (self-attention) ([2, 4096, 320])
        # q,k,v : head-separated q_in, k_in and v_in
        
        ctx_k, ctx_v = hypernetwork.apply_hypernetwork(
            shared.loaded_hypernetwork,
            context if context is not None else x
        )
        
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
        
        qk: Tensor = rearrange(sim, '(b h) d t -> b h t d', h=module.heads).detach().clone()
        vqk: Tensor = o.detach().clone()
        
        self.log(f"    | q: {shape(q_in)} # {shape(q)}")
        self.log(f"    | k: {shape(k_in)} # {shape(k)}")
        self.log(f"    | v: {shape(v_in)} # {shape(v)}")
        #self.log(f"    | qk: {shape(qk)} # {shape(sim)}")
        #self.log(f"    | vqk: {shape(vqk)}")
        
        del q_in, k_in, v_in, q, k, v, sim, o_in, o
        
        return qk, vqk
    
    def feature_to_grid_images(self, feature: AttnFeatureInfo, layer: str, img_idx: int, step: int, width: int, height: int, color: Colorizer):
        w, h, ch = get_shape(layer, width, height)
        # qk
        qk = feature.qk
        heads_qk, ch_qk, n_qk = qk.shape
        assert ch_qk == 77
        assert w * h == n_qk, f"w={w}, h={h}, n_qk={n_qk}"
        qk1 = rearrange(qk, 'a t (h w) -> (a t) h w', h=h).contiguous()
        # vqk
        vqk = feature.vqk
        n_vqk, ch_vqk = vqk.shape
        assert w * h == n_vqk, f"w={w}, h={h}, n_qk={n_vqk}"
        assert ch == ch_vqk, f"ch={ch}, ch_vqk={ch_vqk}"
        vqk1 = rearrange(vqk, '(h w) c -> c h w', h=h).contiguous()
        
        #print(img_idx, step, layer, qk1.shape, vqk1.shape)
        return tutils.tensor_to_image(qk1, ch_qk, heads_qk, color)
    
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
