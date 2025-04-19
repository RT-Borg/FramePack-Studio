from diffusers_helper.hf_login import login

import os
import json
import re

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


# Add this new data structure to store section-specific prompts
class PromptSection:
    def __init__(self, prompt, start_time=0, end_time=None):
        self.prompt = prompt
        self.start_time = start_time  # in seconds
        self.end_time = end_time  # in seconds, None means until the end


def snap_to_section_boundaries(prompt_sections, latent_window_size, fps=30):
    """Adjust timestamps to align with model's internal section boundaries"""
    section_duration = (latent_window_size * 4 - 3) / fps  # Duration of one section in seconds
    
    aligned_sections = []
    for section in prompt_sections:
        # Snap start time to nearest section boundary
        aligned_start = round(section.start_time / section_duration) * section_duration
        
        # Snap end time to nearest section boundary
        aligned_end = None
        if section.end_time is not None:
            aligned_end = round(section.end_time / section_duration) * section_duration
        
        # Ensure minimum section length
        if aligned_end is not None and aligned_end <= aligned_start:
            aligned_end = aligned_start + section_duration
            
        aligned_sections.append(PromptSection(
            prompt=section.prompt,
            start_time=aligned_start,
            end_time=aligned_end
        ))
    
    return aligned_sections


def parse_timestamped_prompt(prompt_text, total_duration, latent_window_size=9):
    """
    Parse a prompt with timestamps in the format [0s-2s: text] or [3s: text]
    Returns a list of PromptSection objects with timestamps aligned to section boundaries
    and reversed to account for reverse generation
    """
    # Default prompt for the entire duration if no timestamps are found
    if "[" not in prompt_text or "]" not in prompt_text:
        return [PromptSection(prompt=prompt_text.strip())]
    
    sections = []
    # Find all timestamp sections [time: text]
    timestamp_pattern = r'\[(\d+(?:\.\d+)?s)(?:-(\d+(?:\.\d+)?s))?\s*:\s*(.*?)\]'
    regular_text = prompt_text
    
    for match in re.finditer(timestamp_pattern, prompt_text):
        start_time_str = match.group(1)
        end_time_str = match.group(2)
        section_text = match.group(3).strip()
        
        # Convert time strings to seconds
        start_time = float(start_time_str.rstrip('s'))
        end_time = float(end_time_str.rstrip('s')) if end_time_str else None
        
        sections.append(PromptSection(
            prompt=section_text,
            start_time=start_time,
            end_time=end_time
        ))
        
        # Remove the processed section from regular_text
        regular_text = regular_text.replace(match.group(0), "")
    
    # If there's any text outside of timestamp sections, use it as a default for the entire duration
    regular_text = regular_text.strip()
    if regular_text:
        sections.append(PromptSection(
            prompt=regular_text,
            start_time=0,
            end_time=None
        ))
    
    # Sort sections by start time
    sections.sort(key=lambda x: x.start_time)
    
    # Fill in end times if not specified
    for i in range(len(sections) - 1):
        if sections[i].end_time is None:
            sections[i].end_time = sections[i+1].start_time
    
    # Set the last section's end time to the total duration if not specified
    if sections and sections[-1].end_time is None:
        sections[-1].end_time = total_duration
    
    # Snap timestamps to section boundaries
    sections = snap_to_section_boundaries(sections, latent_window_size)
    
    # Now reverse the timestamps to account for reverse generation
    reversed_sections = []
    for section in sections:
        reversed_start = total_duration - section.end_time if section.end_time is not None else 0
        reversed_end = total_duration - section.start_time
        reversed_sections.append(PromptSection(
            prompt=section.prompt,
            start_time=reversed_start,
            end_time=reversed_end
        ))
    
    # Sort the reversed sections by start time
    reversed_sections.sort(key=lambda x: x.start_time)
    
    return reversed_sections

# Add this function after parse_timestamped_prompt function but before the worker function
@torch.no_grad()
def load_lora(transformer, lora_path, lora_weight=1.0):
    if not lora_path or lora_path.lower() == "none":
        return transformer
    
    # Load LoRA weights
    lora_state_dict = {}
    
    if lora_path.endswith(".safetensors"):
        lora_state_dict = sf.load_file(lora_path)
    else:
        lora_state_dict = torch.load(lora_path, map_location="cpu")
    
    # Get attention processors
    attn_procs = {}
    for name in transformer.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else transformer.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = transformer.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(transformer.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = transformer.config.block_out_channels[block_id]
        
        attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=4,
        )
    
    # Set attention processors
    transformer.set_attn_processor(attn_procs)
    
    # Load LoRA weights with scaling
    lora_layers = AttnProcsLayers(transformer.attn_processors)
    lora_layers.load_state_dict(lora_state_dict)
    
    # Scale weights
    if lora_weight != 1.0:
        for module in lora_layers.modules():
            if isinstance(module, LoRAAttnProcessor):
                module.to_q_lora.weight.data *= lora_weight
                module.to_k_lora.weight.data *= lora_weight
                module.to_v_lora.weight.data *= lora_weight
                module.to_out_lora.weight.data *= lora_weight
    
    return transformer



@torch.no_grad()
def worker(input_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_path, lora_weight):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Parse the timestamped prompt with boundary snapping and reversing
    prompt_sections = parse_timestamped_prompt(prompt_text, total_second_length, latent_window_size)
    
    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Pre-encode all prompts
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding all prompts...'))))
        
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        # Create a dictionary to store encoded prompts
        encoded_prompts = {}
        for section in prompt_sections:
            if section.prompt not in encoded_prompts:
                llama_vec, clip_l_pooler = encode_prompt_conds(
                    section.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                )
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                encoded_prompts[section.prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)
        
        # Encode negative prompt
        if cfg == 1:
            llama_vec_n, llama_attention_mask_n, clip_l_pooler_n = (
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][0]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][1]),
                torch.zeros_like(encoded_prompts[prompt_sections[0].prompt][2])
            )
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        for prompt_key in encoded_prompts:
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[prompt_key]
            llama_vec = llama_vec.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            encoded_prompts[prompt_key] = (llama_vec, llama_attention_mask, clip_l_pooler)
            
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        if lora_path and lora_path != "None":
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Applying LoRA weights...'))))
            transformer = load_lora(transformer, lora_path, lora_weight)

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

           # Calculate current time position to determine which prompt to use
            current_time_position = (total_generated_latent_frames * 4 - 3) / 30  # in seconds
            if current_time_position < 0:
                current_time_position = 0.01

            
            # Find the appropriate prompt for this section
            current_prompt = prompt_sections[0].prompt  # Default to first prompt
            for section in prompt_sections:
                if section.start_time <= current_time_position and (section.end_time is None or current_time_position < section.end_time):
                    current_prompt = section.prompt
                    break
            
            # Get the encoded prompt for this section
            llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[current_prompt]

            # Calculate the original (non-reversed) time position for display
            original_time_position = total_second_length - current_time_position
            if original_time_position < 0:
                original_time_position = 0
                
            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, ' 
                  f'time position: {current_time_position:.2f}s (original: {original_time_position:.2f}s), '
                  f'using prompt: {current_prompt[:30]}...')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                
                # Calculate current time position and original (non-reversed) position
                current_pos = (total_generated_latent_frames * 4 - 3) / 30
                original_pos = total_second_length - current_pos
                if current_pos < 0: current_pos = 0
                if original_pos < 0: original_pos = 0
                
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, ' \
                       f'Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). ' \
                       f'Current position: {current_pos:.2f}s (original: {original_pos:.2f}s). ' \
                       f'Using prompt: "{current_prompt[:50]}..."'
                
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return

def process(input_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_path, lora_weight):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt_text, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_path, lora_weight)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


# Calculate section boundaries for UI display
section_duration = (9 * 4 - 3) / 30  # Using default latent_window_size=9
section_boundaries = ", ".join([f"{i*section_duration:.1f}s" for i in range(10)])  # Show first 10 boundaries

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
    '[0s-2s: The person waves hello] [2s-4s: The person jumps up and down] [4s: The person does a spin]',
    '[0s-2.5s: The person raises both arms slowly] [2.5s: The person claps hands enthusiastically]',
    '[0s-1.1s: Person gives thumbs up] [1.1s-2.2s: Person smiles and winks] [2.2s-3.3s: Person shows two thumbs down]',
    '[0s-1.1s: Person looks surprised] [1.1s-2.2s: Person raises arms above head] [2.2s-3.3s: Person puts hands on hips]'
]
quick_prompts = [[x] for x in quick_prompts]

# Lora directory
lora_dir = "./loras/"
os.makedirs(lora_dir, exist_ok=True)
lora_files = [os.path.join(lora_dir, f) for f in os.listdir(lora_dir) 
              if f.endswith((".safetensors", ".pt", ".bin"))]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack with Timestamped Prompts')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            
            gr.Markdown(f"""
            ### Prompt with Timestamps
            You can use timestamps in your prompt to change the action at specific times:
            - Format: `[0s-2s: person waves]` or `[3s: person jumps]`
            - Example: `[0s-2s: The person waves hello] [2s-4s: The person jumps]`
            
            For best results, align your timestamps with these section boundaries:
            {section_boundaries}...
            
            Write prompts in natural order (beginning to end). The system will automatically handle the reverse generation.
            """)
            
            prompt = gr.Textbox(label="Prompt", value='The girl dances gracefully, with clear movements, full of charm.')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)

                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

            with gr.Group():
                lora_dropdown = gr.Dropdown(choices=["None"] + lora_files, value="None", label="Select LoRA")
                lora_weight = gr.Slider(label="LoRA Weight", minimum=0.0, maximum=2.0, value=1.0, step=0.01)


        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            gr.Markdown("""
            ## Tips for Best Results
            
            1. Keep prompt sections short and clear (10-15 words per section)
            2. Align timestamps with section boundaries for more precise control
            3. Allow at least 1.1 seconds per action for best results
            4. Use simple, descriptive language focusing on visible actions
            5. For complex sequences, use fewer sections with longer durations
            """)
    
    # Connect the main process

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, lora_dropdown, lora_weight]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])

    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)