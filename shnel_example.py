# conda create -n flux python=3.11
# conda activate flux
# pip install torch==2.3.1
# pip install diffusers==0.30.0 transformers==4.43.3
# pip install sentencepiece==0.2.0 accelerate==0.33.0 protobuf==5.27.3

import torch
from diffusers import  FluxPipeline
import diffusers

_flux_rope = diffusers.models.transformers.transformer_flux.rope
def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    return _flux_rope(pos, dim, theta)

diffusers.models.transformers.transformer_flux.rope = new_flux_rope

pipe = FluxPipeline.from_pretrained("weights/FLUX.1-schnell", revision='refs/pr/1',  torch_dtype=torch.bfloat16)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16) 

prompt = "A cat holding a sign that says hello world"
out = pipe(
     prompt=prompt,
     guidance_scale=0.,
     height=1024,
     width=1024,
     num_inference_steps=4,
     max_sequence_length=256,
).images[0]
out.save("flux_image.png")