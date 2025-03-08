import gc
import logging
import os
import subprocess
import platform
import time
import base64
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
    Returns a filename in target_dir with numbering (e.g. 0001.mp4 or 0001.mp3) that does not yet exist.
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
                            cfg_strength: float, duration: float, generations: int, save_params: bool = True):
    results = []
    video_info = load_video(video, duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = video_info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    start_time = time.time()
    total = generations
    for i in range(generations):
        iter_start = time.time()
        rng = torch.Generator(device=device)
        # If seed is -1, generate a new random seed; else add generation index
        if seed == -1:
            local_seed = torch.seed()
        else:
            local_seed = seed + i
        rng.manual_seed(local_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
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
        if save_params:
            params_content = (
                f"Generation Type: Video-to-Audio\n"
                f"Input Video: {video}\n"
                f"Prompt: {prompt}\n"
                f"Negative Prompt: {negative_prompt}\n"
                f"Used Seed: {local_seed}\n"
                f"Num Steps: {num_steps}\n"
                f"Guidance Strength: {cfg_strength}\n"
                f"Duration (sec): {duration}\n"
                f"Generation: {i+1} out of {generations}\n"
                f"Timestamp: {datetime.now()}\n"
            )
            params_filepath = output_path.with_name(f"{output_path.stem}_Params.txt")
            with open(params_filepath, "w") as pf:
                pf.write(params_content)
        results.append(str(output_path))
        gc.collect()

        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        processed = i + 1
        avg_time = elapsed / processed
        remain = total - processed
        eta = avg_time * remain
        print(f"{processed}/{total} Video-to-Audio generation completed. "
              f"Generation took {iter_time:.2f}s, avg: {avg_time:.2f}s, ETA: {eta:.2f}s.")
    return results

@torch.inference_mode()
def text_to_audio_single(prompt: str, negative_prompt: str, seed: int, num_steps: int,
                           cfg_strength: float, duration: float, generations: int, output_folder: Path = output_dir, save_params: bool = True):
    results = []
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    start_time = time.time()
    total = generations
    for i in range(generations):
        iter_start = time.time()
        rng = torch.Generator(device=device)
        if seed == -1:
            local_seed = torch.seed()
        else:
            local_seed = seed + i
        rng.manual_seed(local_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
        audios = generate(None,
                          None, [prompt],
                          negative_text=[negative_prompt],
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]
        output_path = get_next_numbered_filename(output_folder, "mp3")
        if audio.dim() == 2 and audio.shape[0] == 1:
            audio_stereo = torch.cat([audio, audio], dim=0)
            torchaudio.save(str(output_path), audio_stereo, seq_cfg.sampling_rate)
        else:
            torchaudio.save(str(output_path), audio, seq_cfg.sampling_rate)
        if save_params:
            params_content = (
                f"Generation Type: Text-to-Audio\n"
                f"Prompt: {prompt}\n"
                f"Negative Prompt: {negative_prompt}\n"
                f"Used Seed: {local_seed}\n"
                f"Num Steps: {num_steps}\n"
                f"Guidance Strength: {cfg_strength}\n"
                f"Duration (sec): {duration}\n"
                f"Generation: {i+1} out of {generations}\n"
                f"Timestamp: {datetime.now()}\n"
            )
            params_filepath = output_path.with_name(f"{output_path.stem}_Params.txt")
            with open(params_filepath, "w") as pf:
                pf.write(params_content)
        results.append(str(output_path))
        gc.collect()

        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        processed = i + 1
        avg_time = elapsed / processed
        remain = total - processed
        eta = avg_time * remain
        print(f"{processed}/{total} Text-to-Audio generation completed. "
              f"Generation took {iter_time:.2f}s, avg: {avg_time:.2f}s, ETA: {eta:.2f}s.")
    return results

@torch.inference_mode()
def image_to_audio_single(image, prompt: str, negative_prompt: str, seed: int, num_steps: int,
                            cfg_strength: float, duration: float, generations: int, save_params: bool = True):
    results = []
    image_info = load_image(image)
    clip_frames = image_info.clip_frames.unsqueeze(0)
    sync_frames = image_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    # Fix image proportions: crop frames so height and width are even
    frame_tensor = clip_frames
    _, H, W = frame_tensor[0, 0].shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    if new_H != H or new_W != W:
        clip_frames = clip_frames[:, :, :, :new_H, :new_W]
        sync_frames = sync_frames[:, :, :, :new_H, :new_W]

    start_time = time.time()
    total = generations
    for i in range(generations):
        iter_start = time.time()
        rng = torch.Generator(device=device)
        if seed == -1:
            local_seed = torch.seed()
        else:
            local_seed = seed + i
        rng.manual_seed(local_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
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
        video_info_local = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
        if hasattr(video_info_local, 'clip_frames'):
            frames = video_info_local.clip_frames
            _, C, H, W = frames.shape
            new_H = H if H % 2 == 0 else H - 1
            new_W = W if W % 2 == 0 else W - 1
            if new_H != H or new_W != W:
                video_info_local.clip_frames = frames[:, :, :new_H, :new_W]
        make_video(video_info_local, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
        if save_params:
            params_content = (
                f"Generation Type: Image-to-Audio (experimental)\n"
                f"Input Image: {image}\n"
                f"Prompt: {prompt}\n"
                f"Negative Prompt: {negative_prompt}\n"
                f"Used Seed: {local_seed}\n"
                f"Num Steps: {num_steps}\n"
                f"Guidance Strength: {cfg_strength}\n"
                f"Duration (sec): {duration}\n"
                f"Generation: {i+1} out of {generations}\n"
                f"Timestamp: {datetime.now()}\n"
            )
            params_filepath = output_path.with_name(f"{output_path.stem}_Params.txt")
            with open(params_filepath, "w") as pf:
                pf.write(params_content)
        results.append(str(output_path))
        gc.collect()

        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        processed = i + 1
        avg_time = elapsed / processed
        remain = total - processed
        eta = avg_time * remain
        print(f"{processed}/{total} Image-to-Audio generation completed. "
              f"Generation took {iter_time:.2f}s, avg: {avg_time:.2f}s, ETA: {eta:.2f}s.")
    return results

# --- Wrapper functions for single processing to also return a status message ---

def video_to_audio_single_wrapper(video, prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, save_params):
    results = video_to_audio_single(video, prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, save_params)
    return results, "Done"

def text_to_audio_single_wrapper(prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, save_params):
    results = text_to_audio_single(prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, output_folder=output_dir, save_params=save_params)
    html_output = ""
    for file_path in results:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        html_output += f'<div style="margin-bottom:10px;"><audio controls src="data:audio/mp3;base64,{b64}" style="width:100%;"></audio></div>'
    return html_output, "Done"

def image_to_audio_single_wrapper(image, prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, save_params):
    results = image_to_audio_single(image, prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, save_params)
    return results, "Done"

# --------------------------
# Batch Processing Functions
# --------------------------

@torch.inference_mode()
def batch_video_to_audio(video_path: str, prompt: str, negative_prompt: str, seed: int,
                           num_steps: int, cfg_strength: float, duration: float, generations: int, output_folder: Path, save_params: bool):
    video_info = load_video(video_path, duration)
    clip_frames = video_info.clip_frames.unsqueeze(0)
    sync_frames = video_info.sync_frames.unsqueeze(0)
    seq_cfg.duration = video_info.duration_sec
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    results = []
    start_time = time.time()
    total = generations
    for i in range(generations):
        iter_start = time.time()
        rng = torch.Generator(device=device)
        if seed == -1:
            local_seed = torch.seed()
        else:
            local_seed = seed + i
        rng.manual_seed(local_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
        audios = generate(clip_frames,
                          sync_frames, [prompt],
                          negative_text=[negative_prompt],
                          feature_utils=feature_utils,
                          net=net,
                          fm=fm,
                          rng=rng,
                          cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]
        base_name = Path(video_path).stem
        ext = ".mp4"
        if generations == 1:
            out_filename = Path(video_path).name
        else:
            out_filename = f"{base_name}_{i:02d}{ext}"
        output_path = output_folder / out_filename
        make_video(video_info, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
        if save_params:
            params_content = (
                f"Generation Type: Video-to-Audio (Batch)\n"
                f"Input Video: {video_path}\n"
                f"Prompt: {prompt}\n"
                f"Negative Prompt: {negative_prompt}\n"
                f"Used Seed: {local_seed}\n"
                f"Num Steps: {num_steps}\n"
                f"Guidance Strength: {cfg_strength}\n"
                f"Duration (sec): {duration}\n"
                f"Generation: {i+1} out of {generations}\n"
                f"Timestamp: {datetime.now()}\n"
            )
            params_filepath = output_path.with_name(f"{output_path.stem}_Params.txt")
            with open(params_filepath, "w") as pf:
                pf.write(params_content)
        results.append(str(output_path))

        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        processed = i + 1
        avg_time = elapsed / processed
        remain = total - processed
        eta = avg_time * remain
        print(f"File {video_path}: Generation {processed}/{total} completed. "
              f"Generation took {iter_time:.2f}s, avg: {avg_time:.2f}s, ETA: {eta:.2f}s.")
    gc.collect()
    return results

@torch.inference_mode()
def batch_image_to_audio(image_path: str, prompt: str, negative_prompt: str, seed: int,
                           num_steps: int, cfg_strength: float, duration: float, generations: int, output_folder: Path, save_params: bool):
    image_info = load_image(image_path)
    clip_frames = image_info.clip_frames.unsqueeze(0)
    sync_frames = image_info.sync_frames.unsqueeze(0)
    frame_tensor = clip_frames
    _, H, W = frame_tensor[0, 0].shape
    new_H = H if H % 2 == 0 else H - 1
    new_W = W if W % 2 == 0 else W - 1
    if new_H != H or new_W != W:
        clip_frames = clip_frames[:, :, :, :new_H, :new_W]
        sync_frames = sync_frames[:, :, :, :new_H, :new_W]
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    results = []
    start_time = time.time()
    total = generations
    for i in range(generations):
        iter_start = time.time()
        rng = torch.Generator(device=device)
        if seed == -1:
            local_seed = torch.seed()
        else:
            local_seed = seed + i
        rng.manual_seed(local_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=int(num_steps))
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
        base_name = Path(image_path).stem
        if generations == 1:
            out_filename = base_name + ".mp4"
        else:
            out_filename = f"{base_name}_{i:02d}.mp4"
        output_path = output_folder / out_filename
        video_info_local = VideoInfo.from_image_info(image_info, duration, fps=Fraction(1))
        if hasattr(video_info_local, 'clip_frames'):
            frames = video_info_local.clip_frames
            _, C, H, W = frames.shape
            new_H = H if H % 2 == 0 else H - 1
            new_W = W if W % 2 == 0 else W - 1
            if new_H != H or new_W != W:
                video_info_local.clip_frames = frames[:, :, :new_H, :new_W]
        make_video(video_info_local, output_path, audio, sampling_rate=seq_cfg.sampling_rate)
        if save_params:
            params_content = (
                f"Generation Type: Image-to-Audio (Batch)\n"
                f"Input Image: {image_path}\n"
                f"Prompt: {prompt}\n"
                f"Negative Prompt: {negative_prompt}\n"
                f"Used Seed: {local_seed}\n"
                f"Num Steps: {num_steps}\n"
                f"Guidance Strength: {cfg_strength}\n"
                f"Duration (sec): {duration}\n"
                f"Generation: {i+1} out of {generations}\n"
                f"Timestamp: {datetime.now()}\n"
            )
            params_filepath = output_path.with_name(f"{output_path.stem}_Params.txt")
            with open(params_filepath, "w") as pf:
                pf.write(params_content)
        results.append(str(output_path))

        elapsed = time.time() - start_time
        iter_time = time.time() - iter_start
        processed = i + 1
        avg_time = elapsed / processed
        remain = total - processed
        eta = avg_time * remain
        print(f"File {image_path}: Generation {processed}/{total} completed. "
              f"Generation took {iter_time:.2f}s, avg: {avg_time:.2f}s, ETA: {eta:.2f}s.")
    gc.collect()
    return results

def batch_video_processing_callback(batch_in_folder: str, batch_out_folder: str, skip_existing: bool,
                                    prompt: str, negative_prompt: str, seed: int, num_steps: int,
                                    cfg_strength: float, duration: float, generations: int, save_params: bool):
    global cancel_batch_video
    cancel_batch_video = False
    in_path = Path(batch_in_folder)
    out_path = Path(batch_out_folder)
    out_path.mkdir(exist_ok=True, parents=True)
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.mpeg'}
    files = [f for f in in_path.iterdir() if f.suffix.lower() in video_exts and f.is_file()]
    total_files = len(files)
    total_tasks = total_files * generations
    processed_global = 0
    log_lines = []
    start_time_global = time.time()
    if total_files == 0:
        yield "No video files found in the input folder."
        return
    for f in files:
        if cancel_batch_video:
            log_lines.append("Batch processing cancelled.")
            yield "\n".join(log_lines)
            return
        # Check for a .txt file with the same base name to override the prompt
        txt_file = f.with_suffix(".txt")
        effective_prompt = prompt
        if txt_file.exists():
            with open(txt_file, 'r') as tf:
                content = tf.read().strip()
            if content:
                effective_prompt = content
        dest = out_path / f.name
        if skip_existing and dest.exists():
            log_lines.append(f"Skipping {f.name} (already exists).")
            processed_global += generations
            yield "\n".join(log_lines)
            continue
        try:
            results = batch_video_to_audio(str(f), effective_prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, out_path, save_params)
            processed_global += len(results)
            elapsed_global = time.time() - start_time_global
            avg_time_global = elapsed_global / processed_global if processed_global > 0 else 0
            remain_global = total_tasks - processed_global
            eta_global = avg_time_global * remain_global if processed_global > 0 else 0
            log_lines.append(f"Overall progress: Processed {f.name} with {generations} generation(s) "
                             f"({processed_global}/{total_tasks}). Elapsed: {elapsed_global:.2f}s, "
                             f"avg: {avg_time_global:.2f}s, ETA: {eta_global:.2f}s.")
            yield "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"Error processing {f.name}: {str(e)}")
            yield "\n".join(log_lines)
    yield "\n".join(log_lines)

def batch_image_processing_callback(batch_in_folder: str, batch_out_folder: str, skip_existing: bool,
                                    prompt: str, negative_prompt: str, seed: int, num_steps: int,
                                    cfg_strength: float, duration: float, generations: int, save_params: bool):
    global cancel_batch_image
    cancel_batch_image = False
    in_path = Path(batch_in_folder)
    out_path = Path(batch_out_folder)
    out_path.mkdir(exist_ok=True, parents=True)
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    files = [f for f in in_path.iterdir() if f.suffix.lower() in image_exts and f.is_file()]
    total_files = len(files)
    total_tasks = total_files * generations
    processed_global = 0
    log_lines = []
    start_time_global = time.time()
    if total_files == 0:
        yield "No image files found in the input folder."
        return
    for f in files:
        if cancel_batch_image:
            log_lines.append("Batch processing cancelled.")
            yield "\n".join(log_lines)
            return
        # Check for a .txt file with the same base name to override the prompt
        txt_file = f.with_suffix(".txt")
        effective_prompt = prompt
        if txt_file.exists():
            with open(txt_file, 'r') as tf:
                content = tf.read().strip()
            if content:
                effective_prompt = content
        base_name = Path(f).stem
        if skip_existing:
            out_filename = base_name + (".mp4" if generations == 1 else f"_{0:02d}.mp4")
            dest = out_path / out_filename
            if dest.exists():
                log_lines.append(f"Skipping {f.name} (output exists).")
                processed_global += generations
                yield "\n".join(log_lines)
                continue
        try:
            results = batch_image_to_audio(str(f), effective_prompt, negative_prompt, seed, num_steps, cfg_strength, duration, generations, out_path, save_params)
            processed_global += len(results)
            elapsed_global = time.time() - start_time_global
            avg_time_global = elapsed_global / processed_global if processed_global else 0
            remain_global = total_tasks - processed_global
            eta_global = avg_time_global * remain_global if processed_global else 0
            log_lines.append(f"Overall progress: Processed {f.name} with {generations} generation(s) "
                             f"({processed_global}/{total_tasks}). Elapsed: {elapsed_global:.2f}s, "
                             f"avg: {avg_time_global:.2f}s, ETA: {eta_global:.2f}s")
            yield "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"Error processing {f.name}: {str(e)}")
            yield "\n".join(log_lines)
    yield "\n".join(log_lines)

def batch_text_processing_callback(batch_prompts: str, negative_prompt: str, seed: int, num_steps: int,
                                   cfg_strength: float, duration: float, generations: int, batch_out_folder: str, save_params: bool):
    global cancel_batch_text
    cancel_batch_text = False
    lines = batch_prompts.splitlines()
    total_tasks = len(lines) * generations
    processed_global = 0
    log_lines = []
    start_time = time.time()
    batch_out_folder_path = Path(batch_out_folder)
    batch_out_folder_path.mkdir(exist_ok=True, parents=True)
    if len(lines) == 0:
        yield "No prompts found."
        return
    for line in lines:
        if cancel_batch_text:
            log_lines.append("Batch processing cancelled.")
            yield "\n".join(log_lines)
            return
        prompt_line = line.strip()
        if len(prompt_line) < 2:
            log_lines.append(f"Skipping prompt '{line}' (too short).")
            yield "\n".join(log_lines)
            continue
        try:
            results = text_to_audio_single(prompt_line, negative_prompt, seed, num_steps, cfg_strength, duration,
                                           generations, output_folder=batch_out_folder_path, save_params=save_params)
            processed_global += generations
            elapsed_global = time.time() - start_time
            avg_time_global = elapsed_global / processed_global if processed_global else 0
            remain_global = total_tasks - processed_global
            eta_global = avg_time_global * remain_global if processed_global else 0
            log_lines.append(f"Processed prompt '{prompt_line}' with {generations} generation(s) "
                             f"({processed_global}/{total_tasks}). Elapsed: {elapsed_global:.2f}s, "
                             f"avg: {avg_time_global:.2f}s, ETA: {eta_global:.2f}s")
            yield "\n".join(log_lines)
        except Exception as e:
            log_lines.append(f"Error processing prompt '{prompt_line}': {str(e)}")
            yield "\n".join(log_lines)
    yield "\n".join(log_lines)

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
    # Clear video input, prompt, negative prompt; return default slider values and reset save checkbox to True.
    return None, "", "music", -1, 50, 4.5, 8, 1, True

def clear_text_inputs():
    return "", "", -1, 50, 4.5, 8, 1, True

def clear_image_inputs():
    return None, "", "", -1, 50, 4.5, 8, 1, True

# --------------------------
# Gradio Interface – Using Blocks
# --------------------------

with gr.Blocks() as demo:
    gr.Markdown("# MMAudio SECourses APP V3 : https://www.patreon.com/posts/117990364")
    with gr.Tabs():
        # ---------------- Video-to-Audio Tab ----------------
        with gr.TabItem("Video-to-Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Video Input", height=512)
                    with gr.Row():
                        clear_btn_video = gr.Button("Clear")
                        submit_btn_video = gr.Button("Submit", variant="primary")
                    prompt_video = gr.Textbox(label="Prompt")
                    neg_prompt_video = gr.Textbox(label="Negative prompt", value="music")
                    with gr.Row():
                        seed_slider_video = gr.Slider(label="Seed (-1: random)", minimum=-1, maximum=2147483647, step=1, value=-1, interactive=True)
                        gen_slider_video = gr.Slider(label="Number of Generations", minimum=1, maximum=20, step=1, value=1, interactive=True)
                        steps_slider_video = gr.Slider(label="Num steps", minimum=10, maximum=100, step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_video = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10, step=0.1, value=4.5, interactive=True)
                        duration_slider_video = gr.Slider(label="Duration (sec)", minimum=1, maximum=30, step=1, value=5, interactive=True)
                    # Save Gen Params checkbox for single processing (default enabled)
                    save_params_video = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                with gr.Column(scale=1):
                    output_videos = gr.Gallery(label="Output Videos", show_label=True, elem_id="output_videos")
                    status_video = gr.Markdown(label="Status", value="")
                    open_outputs_btn_video = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_input_videos = gr.Textbox(label="Batch Input Videos Folder Path")
                    batch_output_videos = gr.Textbox(label="Batch Output Videos Folder Path", value=str(output_dir))
                    skip_checkbox_video = gr.Checkbox(label="Skip if existing", value=True)
                    # Save Gen Params checkbox for batch processing (default enabled)
                    batch_save_params_video = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                    with gr.Row():
                        batch_start_video = gr.Button("Start Batch Processing", variant="primary")
                        batch_cancel_video = gr.Button("Cancel Batch Processing")
                    batch_status_video = gr.Markdown(label="Batch Status", value="")
            clear_btn_video.click(fn=clear_video_inputs,
                                  outputs=[video_input, prompt_video, neg_prompt_video,
                                           seed_slider_video, steps_slider_video,
                                           guidance_slider_video, duration_slider_video, gen_slider_video, save_params_video])
            submit_btn_video.click(lambda: ([], "Processing started..."),
                                   outputs=[output_videos, status_video])\
                .then(video_to_audio_single_wrapper,
                      inputs=[video_input, prompt_video, neg_prompt_video, seed_slider_video, steps_slider_video,
                              guidance_slider_video, duration_slider_video, gen_slider_video, save_params_video],
                      outputs=[output_videos, status_video])
            open_outputs_btn_video.click(fn=open_outputs_folder, outputs=[])
            batch_start_video.click(lambda: "Processing started...", outputs=batch_status_video)\
                .then(batch_video_processing_callback,
                      inputs=[batch_input_videos, batch_output_videos, skip_checkbox_video,
                              prompt_video, neg_prompt_video, seed_slider_video, steps_slider_video,
                              guidance_slider_video, duration_slider_video, gen_slider_video, batch_save_params_video],
                      outputs=batch_status_video)
            batch_cancel_video.click(fn=cancel_batch_video_func, outputs=batch_status_video)
        # ---------------- Text-to-Audio Tab ----------------
        with gr.TabItem("Text-to-Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        clear_btn_text = gr.Button("Clear")
                        submit_btn_text = gr.Button("Submit", variant="primary")
                    prompt_text = gr.Textbox(label="Prompt")
                    neg_prompt_text = gr.Textbox(label="Negative prompt")
                    with gr.Row():
                        seed_slider_text = gr.Slider(label="Seed (-1: random)", minimum=-1, maximum=2147483647, step=1, value=-1, interactive=True)
                        gen_slider_text = gr.Slider(label="Number of Generations", minimum=1, maximum=20, step=1, value=1, interactive=True)
                        steps_slider_text = gr.Slider(label="Num steps", minimum=10, maximum=100, step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_text = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10, step=0.1, value=4.5, interactive=True)
                        duration_slider_text = gr.Slider(label="Duration (sec)", minimum=1, maximum=30, step=1, value=5, interactive=True)
                    save_params_text = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                with gr.Column(scale=1):
                    output_audios_html = gr.HTML(label="Output Audios")
                    status_text = gr.Markdown(label="Status", value="")
                    open_outputs_btn_text = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_prompts = gr.Textbox(label="Batch Prompts (one per line)", lines=5)
                    batch_output_text = gr.Textbox(label="Batch Output Audios Folder Path", value=str(output_dir))
                    batch_save_params_text = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                    with gr.Row():
                        batch_start_text = gr.Button("Start Batch Processing", variant="primary")
                        batch_cancel_text = gr.Button("Cancel Batch Processing")
                    batch_status_text = gr.Markdown(label="Batch Status", value="")
            clear_btn_text.click(fn=clear_text_inputs,
                                 outputs=[prompt_text, neg_prompt_text,
                                          seed_slider_text, steps_slider_text,
                                          guidance_slider_text, duration_slider_text, gen_slider_text, save_params_text])
            submit_btn_text.click(lambda: ("", "Processing started..."),
                                  outputs=[output_audios_html, status_text])\
                .then(text_to_audio_single_wrapper,
                      inputs=[prompt_text, neg_prompt_text, seed_slider_text, steps_slider_text,
                              guidance_slider_text, duration_slider_text, gen_slider_text, save_params_text],
                      outputs=[output_audios_html, status_text])
            open_outputs_btn_text.click(fn=open_outputs_folder, outputs=[])
            batch_start_text.click(lambda: "Processing started...", outputs=batch_status_text)\
                .then(batch_text_processing_callback,
                      inputs=[batch_prompts, neg_prompt_text, seed_slider_text, steps_slider_text,
                              guidance_slider_text, duration_slider_text, gen_slider_text, batch_output_text, batch_save_params_text],
                      outputs=batch_status_text)
            batch_cancel_text.click(fn=cancel_batch_text_func, outputs=batch_status_text)
        # ---------------- Image-to-Audio (experimental) Tab ----------------
        with gr.TabItem("Image-to-Audio (experimental)"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="filepath", label="Image Input", height=512)
                    with gr.Row():
                        clear_btn_image = gr.Button("Clear")
                        submit_btn_image = gr.Button("Submit", variant="primary")
                    prompt_image = gr.Textbox(label="Prompt")
                    neg_prompt_image = gr.Textbox(label="Negative prompt")
                    with gr.Row():
                        seed_slider_image = gr.Slider(label="Seed (-1: random)", minimum=-1, maximum=2147483647, step=1, value=-1, interactive=True)
                        gen_slider_image = gr.Slider(label="Number of Generations", minimum=1, maximum=20, step=1, value=1, interactive=True)
                        steps_slider_image = gr.Slider(label="Num steps", minimum=10, maximum=100, step=1, value=50, interactive=True)
                    with gr.Row():
                        guidance_slider_image = gr.Slider(label="Guidance Strength", minimum=1.5, maximum=10, step=0.1, value=4.5, interactive=True)
                        duration_slider_image = gr.Slider(label="Duration (sec)", minimum=1, maximum=30, step=1, value=5, interactive=True)
                    save_params_image = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                with gr.Column(scale=1):
                    output_videos_image = gr.Gallery(label="Output Videos", show_label=True, elem_id="output_videos_image")
                    status_image = gr.Markdown(label="Status", value="")
                    open_outputs_btn_image = gr.Button("Open Outputs Folder")
                    gr.Markdown("**Batch Processing**")
                    batch_input_images = gr.Textbox(label="Batch Input Images Folder Path")
                    batch_output_images = gr.Textbox(label="Batch Output Videos Folder Path", value=str(output_dir))
                    skip_checkbox_image = gr.Checkbox(label="Skip if existing", value=True)
                    batch_save_params_image = gr.Checkbox(label="Save Gen Params", value=True, interactive=True)
                    with gr.Row():
                        batch_start_image = gr.Button("Start Batch Processing", variant="primary")
                        batch_cancel_image = gr.Button("Cancel Batch Processing")
                    batch_status_image = gr.Markdown(label="Batch Status", value="")
            clear_btn_image.click(fn=clear_image_inputs,
                                  outputs=[image_input, prompt_image, neg_prompt_image,
                                           seed_slider_image, steps_slider_image,
                                           guidance_slider_image, duration_slider_image, gen_slider_image, save_params_image])
            submit_btn_image.click(lambda: ([], "Processing started..."),
                                   outputs=[output_videos_image, status_image])\
                .then(image_to_audio_single_wrapper,
                      inputs=[image_input, prompt_image, neg_prompt_image, seed_slider_image, steps_slider_image,
                              guidance_slider_image, duration_slider_image, gen_slider_image, save_params_image],
                      outputs=[output_videos_image, status_image])
            open_outputs_btn_image.click(fn=open_outputs_folder, outputs=[])
            batch_start_image.click(lambda: "Processing started...", outputs=batch_status_image)\
                .then(batch_image_processing_callback,
                      inputs=[batch_input_images, batch_output_images, skip_checkbox_image,
                              prompt_image, neg_prompt_image, seed_slider_image, steps_slider_image,
                              guidance_slider_image, duration_slider_image, gen_slider_image, batch_save_params_image],
                      outputs=batch_status_image)
            batch_cancel_image.click(fn=cancel_batch_image_func, outputs=batch_status_image)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--share', action='store_true', help='Share Gradio app')
    args = parser.parse_args()
    demo.launch(inbrowser=True, share=args.share, allowed_paths=[str(output_dir)])