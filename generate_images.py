# import sys
# import os
#
# # Add src directory to Python path
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
# # or directly
# sys.path.append("src")
#
# import torch
# from diffusers import StableDiffusionPipeline
# from omegaconf import OmegaConf
# from utils_model import load_model_from_config
#
# if not os.path.exists("generated_images"):
#     os.makedirs("generated_images")
#
#
# def setup_generation():
#     device = torch.device("cuda")
#
#     # Load LDM model
#     ldm_config = "configs/stable-diffusion/v1-inference.yaml"
#     ldm_ckpt = "models/v1-5-pruned.ckpt"  # Base SD checkpoint
#
#     print(f'Building LDM model with config {ldm_config}...')
#     config = OmegaConf.load(ldm_config)
#     ldm_ae = load_model_from_config(config, ldm_ckpt)
#     ldm_aef = ldm_ae.first_stage_model
#     ldm_aef.eval()
#
#     # Load finetuned decoder weights
#     decoder_path = "output/checkpoint_000.pth"
#     state_dict = torch.load(decoder_path)['ldm_decoder']
#     ldm_aef.load_state_dict(state_dict, strict=False)
#
#     # Setup pipeline with memory optimizations
#     pipe = StableDiffusionPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-2",
#         torch_dtype=torch.float16,
#         revision="fp16"
#     ).to(device)
#
#     pipe.enable_attention_slicing()
#     pipe.enable_sequential_cpu_offload()
#
#     # Set the decoder
#     pipe.vae.decode = lambda x, *args, **kwargs: ldm_aef.decode(x).unsqueeze(0)
#
#     return pipe
#
#
# # def setup_generation():
# #     device = torch.device("cuda")
# #
# #     # Load LDM model
# #     ldm_config = "configs/stable-diffusion/v1-inference.yaml"
# #     ldm_ckpt = "models/v1-5-pruned.ckpt"  # Base SD checkpoint
# #
# #     print(f'Building LDM model with config {ldm_config}...')
# #     config = OmegaConf.load(ldm_config)
# #     ldm_ae = load_model_from_config(config, ldm_ckpt)
# #     ldm_aef = ldm_ae.first_stage_model
# #     ldm_aef.eval()
# #
# #     # Load finetuned decoder weights
# #     decoder_path = "output/checkpoint_000.pth"
# #     state_dict = torch.load(decoder_path)['ldm_decoder']
# #     ldm_aef.load_state_dict(state_dict, strict=False)
# #
# #     # Setup pipeline with memory optimizations
# #     pipe = StableDiffusionPipeline.from_pretrained(
# #         "stabilityai/stable-diffusion-2",
# #         torch_dtype=torch.float16,
# #         revision="fp16"
# #     ).to(device)
# #
# #     pipe.enable_attention_slicing()
# #     pipe.enable_sequential_cpu_offload()
# #
# #     # Set the decoder
# #     pipe.vae.decode = lambda x, *args, **kwargs: ldm_aef.decode(x).unsqueeze(0)
# #
# #     return pipe
#
#
# def generate_images(pipe, prompts, output_dir="generated_images"):
#     os.makedirs(output_dir, exist_ok=True)
#
#     for i, prompt in enumerate(prompts):
#         with torch.autocast("cuda"):
#             image = pipe(
#                 prompt,
#                 num_inference_steps=50,
#                 guidance_scale=7.5
#             ).images[0]
#
#         image.save(f"{output_dir}/generated_{i}.png")
#         print(f"Generated image {i} from prompt: {prompt}")
#
#
# # Test prompts
# test_prompts = [
#     "a beautiful sunset over mountains",
#     "a cat playing with yarn",
#     "a futuristic city skyline at night",
#     "an oil painting of flowers in a vase",
#     "a professional photograph of a beach",
#     "a watercolor landscape with misty mountains",
#     "an intricate steampunk-inspired mechanical device",
#     "a serene zen garden with carefully placed rocks",
#     "a vibrant street market in a bustling city",
#     "a detailed portrait of an elderly person with expressive wrinkles",
#     "an abstract representation of musical harmony",
#     "a cyberpunk alleyway with neon lighting",
#     "a surreal desert scene with melting clocks",
#     "a traditional Japanese woodblock print of a river scene",
#     "a hyper-realistic close-up of a butterfly wing",
#     "a minimalist architectural interior",
#     "an underwater coral reef ecosystem",
#     "a nostalgic vintage photograph of a small town",
#     "a cosmic space landscape with distant galaxies",
#     "a whimsical illustration of forest creatures"
# ]
#
# if __name__ == "__main__":
#     pipe = setup_generation()
#     generate_images(pipe, test_prompts)


import sys
import os

sys.path.append("src")

import torch
from diffusers import StableDiffusionPipeline
from omegaconf import OmegaConf
from utils_model import load_model_from_config
from dataclasses import dataclass


@dataclass
class AutoencoderKLOutput:
    sample: torch.FloatTensor


def setup_generation():
    device = torch.device("cuda")

    # Load LDM model
    ldm_config = "configs/stable-diffusion/v1-inference.yaml"
    ldm_ckpt = "models/v1-5-pruned.ckpt"

    print(f'Building LDM model with config {ldm_config}...')
    config = OmegaConf.load(ldm_config)
    ldm_ae = load_model_from_config(config, ldm_ckpt)
    ldm_aef = ldm_ae.first_stage_model
    ldm_aef.eval()

    # Load finetuned decoder weights
    decoder_path = "output/checkpoint_000.pth"
    state_dict = torch.load(decoder_path)['ldm_decoder']
    ldm_aef.load_state_dict(state_dict, strict=False)

    # Setup pipeline with memory optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,
        revision="fp16"
    ).to(device)

    # Modified decoder function to handle dimensions correctly
    def custom_decoder(x, *args, **kwargs):
        x = x.to(device)  # Ensure input is on correct device

        # Decode and ensure correct dimensions [batch_size, channels, height, width]
        decoded = ldm_aef.decode(x)
        if len(decoded.shape) == 3:
            decoded = decoded.unsqueeze(0)
        elif len(decoded.shape) == 5:
            decoded = decoded.squeeze(2)

        # Ensure we have the right shape for the pipeline
        if decoded.shape[1] != 3:  # If channels are not RGB
            decoded = decoded.repeat(1, 3, 1, 1)

        return AutoencoderKLOutput(sample=decoded)

    # Set the decoder
    pipe.vae.decode = custom_decoder
    pipe.enable_attention_slicing()

    return pipe


def generate_images(pipe, prompts, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)

    for i, prompt in enumerate(prompts):
        try:
            with torch.autocast("cuda"):
                image = pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images[0]

            image.save(f"{output_dir}/generate_{i}.png")
            print(f"Generated image {i} from prompt: {prompt}")
        except Exception as e:
            print(f"Error generating image {i}: {str(e)}")
            print(f"Full error: ", e.__class__.__name__)
            import traceback
            traceback.print_exc()


# Test prompts
# test_prompts = [
#     "a beautiful sunset over mountains",
#     "a cat playing with yarn",
#     "a futuristic city skyline at night",
#     "an oil painting of flowers in a vase",
#     "a professional photograph of a beach"
# ]
test_prompts = [
    "a beautiful sunset over mountains",
    "a cat playing with yarn",
    "a futuristic city skyline at night",
    "an oil painting of flowers in a vase",
    "a professional photograph of a beach",
    "a watercolor landscape with misty mountains",
    "an intricate steampunk-inspired mechanical device",
    "a serene zen garden with carefully placed rocks",
    "a vibrant street market in a bustling city",
    "a detailed portrait of an elderly person with expressive wrinkles",
    "an abstract representation of musical harmony",
    "a cyberpunk alleyway with neon lighting",
    "a surreal desert scene with melting clocks",
    "a traditional Japanese woodblock print of a river scene",
    "a hyper-realistic close-up of a butterfly wing",
    "a minimalist architectural interior",
    "an underwater coral reef ecosystem",
    "a nostalgic vintage photograph of a small town",
    "a cosmic space landscape with distant galaxies",
    "a whimsical illustration of forest creatures"
]
if __name__ == "__main__":
    pipe = setup_generation()
    generate_images(pipe, test_prompts)