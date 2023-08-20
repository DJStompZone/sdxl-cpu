import random
import sys
import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_refiner = False
prompt = 'A bright neon sign that says DeepAI, with a dark smoky bar background'
num_imgs = 16

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

if use_refiner:
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner = refiner.to(device)
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

for i in range(num_imgs):
    seed = i + random.randint(0, sys.maxsize)
    print(f"[{i}/{num_imgs}] Prompt:\t{prompt}\nSeed:\t{seed}")
    
    images = pipe(
        prompt=prompt,
        output_type="latent" if use_refiner else "pil",
        generator=torch.Generator(device).manual_seed(seed),
    ).images

    if use_refiner:
        images = refiner(
            prompt=prompt,
            image=images,
        ).images

    images[0].save(f"output_{i}.jpg")
