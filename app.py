import gc
import logging
import os
import subprocess
import platform
import time
from argparse import ArgumentParser
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from mmaudio.eval_utils import (ModelConfig, VideoInfo, all_model_cfg, generate, load_image,
                                load_video, make_video, setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

# Enable TF32 on GPU if available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

# Determine device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    log.warning('CUDA/MPS are not available, running on CPU')
dtype = torch.bfloat16

# Use the pre‐configured “large_44k_v2” model
model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True, parents=True)

setup_eval_logging()

# Global flags for cancelling batch processing
cancel_batch_video = False
cancel_batch_image = False
cancel_batch_text = False

def get_model() -> tuple[MMAudio, FeaturesUtils, SequenceConfig]:
    seq_cfg = model.seq_cfg

    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    return net, feature_utils, model.seq_cfg

net, feature_utils, seq_cfg = get_model()

def get_next_numbered_filename(target_dir: Path, extension: str) -> Path:
    """
    Returns a filename in target_dir with numbering (e.g. 0001.mp4 or 0001.mp3) 
    that does not yet exist.
    """
    i = 1
    while True:
        filename = target_dir / f"{i:04d}.{extension}"
        if not filename.exists():
            return filename
        i += 1

# --------------------------
# Single Processing Functions
# --------------------------

@torch.inference_mode()
def video_to_audio_single(video, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                            cfg_strength: float, duration: float):
    rng = torch.Generator(device=device)
    if seed != -1:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = video_info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    output_path = get_next_numbered_filename(output_dir, "mp4")
    make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return str(output_path)

@torch.inference_mode()
def image_to_audio_single(image, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                            cfg_strength: float, duration: float):
    rng = torch.Generator(device=device)
    if seed != -1:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
    image_info = load_image(image)
    clip_frames = image_info.clip_frames.unsqueeze(0)
    sync_frames = image_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength,
                      image_input=True)
    audio = audios.float().cpu()[0]
    output_path = get_next_numbered_filename(output_dir, "mp4")
    video_info = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
    make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return str(output_path)

@torch.inference_mode()
def text_to_audio_single(prompt: str, negative_prompt: str, seed: int, num_steps: int,
                           cfg_strength: float, duration: float):
    rng = torch.Generator(device=device)
    if seed != -1:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
    clip_frames = sync_frames = None
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    # Save as .mp3 now (numbered) instead of .flac
    output_path = get_next_numbered_filename(output_dir, "mp3")
    # If mono, duplicate channel to create stereo
    if audio.dim() == 2 and audio.shape[0] == 1:
        audio_stereo = torch.cat([audio, audio], dim=0)
        torchaudio.save(str(output_path), audio_stereo, seq_cfg.sampling_rate)
    else:
        torchaudio.save(str(output_path), audio, seq_cfg.sampling_rate)
    gc.collect()
    return str(output_path)

# --------------------------
# Batch Processing Functions
# --------------------------

@torch.inference_mode()
def batch_video_to_audio(video_path: str, prompt: str, negative_prompt: str, seed: int,
                           num_steps: int, cfg_strength: float, duration: float, output_folder: Path):
    rng = torch.Generator(device=device)
    if seed != -1:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
    video_info = load_video(video_path, duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = video_info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength)
    audio = audios.float().cpu()[0]
    # In batch mode, we use the input file's name (do not number)
    output_path = output_folder / Path(video_path).name
    make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return str(output_path)

@torch.inference_mode()
def batch_image_to_audio(image_path: str, prompt: str, negative_prompt: str, seed: int,
                           num_steps: int, cfg_strength: float, duration: float, output_folder: Path):
    rng = torch.Generator(device=device)
    if seed != -1:
        rng.manual_seed(seed)
    else:
        rng.seed()
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
    image_info = load_image(image_path)
    clip_frames = image_info.clip_frames.unsqueeze(0)
    sync_frames = image_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(clip_frames,
                      sync_frames, [prompt],
                      negative_text=[negative_prompt],
                      feature_utils=feature_utils,
                      net=net,
                      fm=fm,
                      rng=rng,
                      cfg_strength=cfg_strength,
                      image_input=True)
    audio = audios.float().cpu()[0]
    # For images, use the base name with .mp4 extension.
    out_filename = Path(image_path).stem + ".mp4"
    output_path = output_folder / out_filename
    video_info = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
    make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
    gc.collect()
    return str(output_path)

def batch_video_processing_callback(batch_in_folder: str, batch_out_folder: str, skip_existing: bool,
                                    prompt: str, negative_prompt: str, seed: int, num_steps: int,
                                    cfg_strength: float, duration: float):
    global cancel_batch_video
    cancel_batch_video = False
    in_path = Path(batch_in_folder)
    out_path = Path(batch_out_folder)
    out_path.mkdir(exist_ok=True, parents=True)
    # Consider common video extensions
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.mpeg'}
    files = [f for f in in_path.iterdir() if f.suffix.lower() in video_exts and f.is_file()]
    total = len(files)
    if total == 0:
        return "No video files found in the input folder."
    processed = 0
    log_lines = []
    start_time = time.time()
    for f in files:
        if cancel_batch_video:
            log_lines.append("Batch processing cancelled.")
            break
        dest = out_path / f.name
        if skip_existing and dest.exists():
            log_lines.append(f"Skipping {f.name} (already exists).")
            continue
        try:
            batch_video_to_audio(str(f), prompt, negative_prompt, seed, num_steps, cfg_strength, duration, out_path)
            processed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed if processed > 0 else 0
            remain = total - processed
            est_time = avg_time * remain
            log_lines.append(f"Processed {f.name} ({processed}/{total}). Elapsed: {elapsed:.1f}s, "
                             f"avg: {avg_time:.1f}s, remaining: {remain}, est. time left: {est_time:.1f}s")
            print(log_lines[-1])
        except Exception as e:
            log_lines.append(f"Error processing {f.name}: {str(e)}")
            print(log_lines[-1])
    return "\n".join(log_lines)

def batch_image_processing_callback(batch_in_folder: str, batch_out_folder: str, skip_existing: bool,
                                    prompt: str, negative_prompt: str, seed: int, num_steps: int,
                                    cfg_strength: float, duration: float):
    global cancel_batch_image
    cancel_batch_image = False
    in_path = Path(batch_in_folder)
    out_path = Path(batch_out_folder)
    out_path.mkdir(exist_ok=True, parents=True)
    # Consider common image extensions
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    files = [f for f in in_path.iterdir() if f.suffix.lower() in image_exts and f.is_file()]
    total = len(files)
    if total == 0:
        return "No image files found in the input folder."
    processed = 0
    log_lines = []
    start_time = time.time()
    for f in files:
        if cancel_batch_image:
            log_lines.append("Batch processing cancelled.")
            break
        # In batch for images, we save as video with the same base name (.mp4)
        out_filename = Path(f).stem + ".mp4"
        dest = out_path / out_filename
        if skip_existing and dest.exists():
            log_lines.append(f"Skipping {f.name} (output exists).")
            continue
        try:
            batch_image_to_audio(str(f), prompt, negative_prompt, seed, num_steps, cfg_strength, duration, out_path)
            processed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed if processed > 0 else 0
            remain = total - processed
            est_time = avg_time * remain
            log_lines.append(f"Processed {f.name} ({processed}/{total}). Elapsed: {elapsed:.1f}s, "
                             f"avg: {avg_time:.1f}s, remaining: {remain}, est. time left: {est_time:.1f}s")
            print(log_lines[-1])
        except Exception as e:
            log_lines.append(f"Error processing {f.name}: {str(e)}")
            print(log_lines[-1])
    return "\n".join(log_lines)

def batch_text_processing_callback(batch_prompts: str, negative_prompt: str, seed: int, num_steps: int,
                                   cfg_strength: float, duration: float):
    global cancel_batch_text
    cancel_batch_text = False
    lines = batch_prompts.splitlines()
    total = len(lines)
    if total == 0:
        return "No prompts found."
    processed = 0
    log_lines = []
    start_time = time.time()
    for line in lines:
        if cancel_batch_text:
            log_lines.append("Batch processing cancelled.")
            break
        prompt = line.strip()
        if len(prompt) < 2:
            log_lines.append(f"Skipping prompt '{line}' (too short).")
            continue
        try:
            text_to_audio_single(prompt, negative_prompt, seed, num_steps, cfg_strength, duration)
            processed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed if processed > 0 else 0
            remain = total - processed
            est_time = avg_time * remain
            log_lines.append(f"Processed prompt '{prompt}' ({processed}/{total}). Elapsed: {elapsed:.1f}s, "
                             f"avg: {avg_time:.1f}s, remaining: {remain}, est. time left: {est_time:.1f}s")
            print(log_lines[-1])
        except Exception as e:
            log_lines.append(f"Error processing prompt '{prompt}': {str(e)}")
            print(log_lines[-1])
    return "\n".join(log_lines)

def cancel_batch_video_func():
    global cancel_batch_video
    cancel_batch_video = True
    return "Batch video processing cancellation requested."

def cancel_batch_image_func():
    global cancel_batch_image
    cancel_batch_image = True
    return "Batch image processing cancellation requested."

def cancel_batch_text_func():
    global cancel_batch_text
    cancel_batch_text = True
    return "Batch text processing cancellation requested."

def open_outputs_folder():
    """Opens the output folder using the system file explorer."""
    p = str(output_dir.resolve())
    if platform.system() == "Windows":
        subprocess.Popen(["explorer", p])
    else:
        subprocess.Popen(["xdg-open", p])
    return "Outputs folder opened."

# --------------------------
# Clear Input Functions
# --------------------------

def clear_video_inputs():
    # Clear video input, prompt, negative prompt; return default slider values.
    return None, "", "music", -1, 50, 4.5, 8

def clear_text_inputs():
    return "", "", -1, 50, 4.5, 8

def clear_image_inputs():
    return None, "", "", -1, 50, 4.5, 8

# --------------------------
# Gradio Interface – Using Blocks
# --------------------------

with gr.Blocks() as demo:
    gr.Markdown("# MMAudio SECourses APP V1 : https://www.patreon.com/posts/117990364")
    with gr.Tabs():
        # ---------------- Video-to-Audio Tab ----------------
        with gr.TabItem("Video-to-Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Video Input",height=512)
                    with gr.Row():
                        clear_btn_video = gr.Button("Clear")
                        submit_btn_video = gr.Button("Submit",variant="primary")
                    prompt_video = gr.Textbox(label="Prompt")
                    neg_prompt_video = gr.Textbox(label="Negative prompt", value="music")
                    with gr.Row():
                        seed_slider_video = gr.Slider(label="Seed (-1: random)", minimum=-1,
                                                      maximum=2147483647, step=1, value=-1, interactive=True)
                        steps_slider_video = gr.Slider(label="Num steps", minimum=10, maximum=100,
                                                       step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_video = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10,
                                                          step=0.1, value=4.5, interactive=True)
                        duration_slider_video = gr.Slider(label="Duration (sec)", minimum=1, maximum=100,
                                                          step=1, value=8, interactive=True)
                with gr.Column(scale=1):
                    output_video = gr.Video(label="Output Video",height=512)
                    open_outputs_btn_video = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_input_videos = gr.Textbox(label="Batch Input Videos Folder Path")
                    batch_output_videos = gr.Textbox(label="Batch Output Videos Folder Path", value=str(output_dir))
                    skip_checkbox_video = gr.Checkbox(label="Skip if existing", value=True)
                    with gr.Row():
                        batch_start_video = gr.Button("Start Batch Processing",variant="primary")
                        batch_cancel_video = gr.Button("Cancel Batch Processing")
                    batch_status_video = gr.Markdown(label="Batch Status")
            # Wire single processing buttons:
            clear_btn_video.click(fn=clear_video_inputs,
                                  outputs=[video_input, prompt_video, neg_prompt_video,
                                           seed_slider_video, steps_slider_video,
                                           guidance_slider_video, duration_slider_video])
            submit_btn_video.click(fn=video_to_audio_single,
                                   inputs=[video_input, prompt_video, neg_prompt_video,
                                           seed_slider_video, steps_slider_video,
                                           guidance_slider_video, duration_slider_video],
                                   outputs=output_video)
            open_outputs_btn_video.click(fn=open_outputs_folder, outputs=[])
            batch_start_video.click(fn=batch_video_processing_callback,
                                    inputs=[batch_input_videos, batch_output_videos, skip_checkbox_video,
                                            prompt_video, neg_prompt_video,
                                            seed_slider_video, steps_slider_video,
                                            guidance_slider_video, duration_slider_video],
                                    outputs=batch_status_video)
            batch_cancel_video.click(fn=cancel_batch_video_func, outputs=batch_status_video)
        # ---------------- Text-to-Audio Tab ----------------
        with gr.TabItem("Text-to-Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        clear_btn_text = gr.Button("Clear")
                        submit_btn_text = gr.Button("Submit",variant="primary")
                    prompt_text = gr.Textbox(label="Prompt")
                    neg_prompt_text = gr.Textbox(label="Negative prompt")
                    with gr.Row():
                        seed_slider_text = gr.Slider(label="Seed (-1: random)", minimum=-1,
                                                     maximum=2147483647, step=1, value=-1, interactive=True)
                        steps_slider_text = gr.Slider(label="Num steps", minimum=10, maximum=100,
                                                      step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_text = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10,
                                                         step=0.1, value=4.5, interactive=True)
                        duration_slider_text = gr.Slider(label="Duration (sec)", minimum=1, maximum=100,
                                                         step=1, value=8, interactive=True)
                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="Output Audio")
                    open_outputs_btn_text = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_prompts = gr.Textbox(label="Batch Prompts (one per line)", lines=5)
                    with gr.Row():
                        batch_start_text = gr.Button("Start Batch Processing",variant="primary")
                        batch_cancel_text = gr.Button("Cancel Batch Processing")
                    batch_status_text = gr.Markdown(label="Batch Status")
            clear_btn_text.click(fn=clear_text_inputs,
                                 outputs=[prompt_text, neg_prompt_text,
                                          seed_slider_text, steps_slider_text,
                                          guidance_slider_text, duration_slider_text])
            submit_btn_text.click(fn=text_to_audio_single,
                                  inputs=[prompt_text, neg_prompt_text,
                                          seed_slider_text, steps_slider_text,
                                          guidance_slider_text, duration_slider_text],
                                  outputs=output_audio)
            open_outputs_btn_text.click(fn=open_outputs_folder, outputs=[])
            batch_start_text.click(fn=batch_text_processing_callback,
                                   inputs=[batch_prompts, neg_prompt_text,
                                           seed_slider_text, steps_slider_text,
                                           guidance_slider_text, duration_slider_text],
                                   outputs=batch_status_text)
            batch_cancel_text.click(fn=cancel_batch_text_func, outputs=batch_status_text)
        # ---------------- Image-to-Audio (experimental) Tab ----------------
        with gr.TabItem("Image-to-Audio (experimental)"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Image Input",height=512)
                    with gr.Row():
                        clear_btn_image = gr.Button("Clear")
                        submit_btn_image = gr.Button("Submit",variant="primary")
                    prompt_image = gr.Textbox(label="Prompt")
                    neg_prompt_image = gr.Textbox(label="Negative prompt")
                    with gr.Row():
                        seed_slider_image = gr.Slider(label="Seed (-1: random)", minimum=-1,
                                                      maximum=2147483647, step=1, value=-1, interactive=True)
                        steps_slider_image = gr.Slider(label="Num steps", minimum=10, maximum=100,
                                                       step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_image = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10,
                                                          step=0.1, value=4.5, interactive=True)
                        duration_slider_image = gr.Slider(label="Duration (sec)", minimum=1, maximum=100,
                                                          step=1, value=8, interactive=True)
                with gr.Column(scale=1):
                    output_video_image = gr.Video(label="Output Video",height=512)
                    open_outputs_btn_image = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_input_images = gr.Textbox(label="Batch Input Images Folder Path")
                    batch_output_images = gr.Textbox(label="Batch Output Videos Folder Path", value=str(output_dir))
                    skip_checkbox_image = gr.Checkbox(label="Skip if existing", value=True)
                    with gr.Row():
                        batch_start_image = gr.Button("Start Batch Processing",variant="primary")
                        batch_cancel_image = gr.Button("Cancel Batch Processing")
                    batch_status_image = gr.Markdown(label="Batch Status")
            clear_btn_image.click(fn=clear_image_inputs,
                                  outputs=[image_input, prompt_image, neg_prompt_image,
                                           seed_slider_image, steps_slider_image,
                                           guidance_slider_image, duration_slider_image])
            submit_btn_image.click(fn=image_to_audio_single,
                                   inputs=[image_input, prompt_image, neg_prompt_image,
                                           seed_slider_image, steps_slider_image,
                                           guidance_slider_image, duration_slider_image],
                                   outputs=output_video_image)
            open_outputs_btn_image.click(fn=open_outputs_folder, outputs=[])
            batch_start_image.click(fn=batch_image_processing_callback,
                                    inputs=[batch_input_images, batch_output_images, skip_checkbox_image,
                                            prompt_image, neg_prompt_image,
                                            seed_slider_image, steps_slider_image,
                                            guidance_slider_image, duration_slider_image],
                                    outputs=batch_status_image)
            batch_cancel_image.click(fn=cancel_batch_image_func, outputs=batch_status_image)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true', help='Share Gradio app')
    args = parser.parse_args()
    demo.launch(inbrowser=True, share=args.share, allowed_paths=[str(output_dir)])