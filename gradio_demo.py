import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import gradio as gr
import torch
import torchaudio
from tqdm import tqdm

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.sequence_config import SequenceConfig
from mmaudio.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

device = 'cuda'
dtype = torch.bfloat16

model: ModelConfig = all_model_cfg['large_44k_v2']
model.download_if_needed()
output_dir = Path('./output/gradio')

setup_eval_logging()

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

    return net, feature_utils, seq_cfg

net, feature_utils, seq_cfg = get_model()

def open_folder(path: Path):
    if os.name == 'nt':
        Popen(['explorer', path])
    else:
        Popen(['xdg-open', path])

@torch.inference_mode()
def video_to_audio(video: gr.Video, prompt: str, negative_prompt: str, random_seed: bool, seed: int,
                   num_steps: int, cfg_strength: float, duration: float, num_generations: int):

    if random_seed:
        seed = random.randint(0, 2**32 - 1)

    seeds = [seed + i for i in range(num_generations)]
    results = []

    for current_seed in tqdm(seeds, desc='Generating videos'):
        rng = torch.Generator(device=device)
        rng.manual_seed(current_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

        video_info = load_video(video, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec
        clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
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

        current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(exist_ok=True, parents=True)
        video_save_path = output_dir / f'{current_time_string}_{current_seed}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        results.append((video_save_path, current_seed))

    return results[0][0], current_seed, gr.update(visible=True)

@torch.inference_mode()
def text_to_audio(prompt: str, negative_prompt: str, random_seed: bool, seed: int, num_steps: int,
                  cfg_strength: float, duration: float, num_generations: int):

    if random_seed:
        seed = random.randint(0, 2**32 - 1)

    seeds = [seed + i for i in range(num_generations)]
    results = []

    for current_seed in tqdm(seeds, desc='Generating audio'):
        rng = torch.Generator(device=device)
        rng.manual_seed(current_seed)
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

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

        current_time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(exist_ok=True, parents=True)
        audio_save_path = output_dir / f'{current_time_string}_{current_seed}.flac'
        torchaudio.save(audio_save_path, audio, seq_cfg.sampling_rate)
        results.append((audio_save_path, current_seed))

    return results[0][0], current_seed, gr.update(visible=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help='share gradio app')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    with gr.Blocks() as demo:
        gr.Markdown("# MMAudio SECourses APP V1 - https://www.patreon.com/posts/117990364")
        with gr.Tab("Video-to-Audio"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Video")
                    prompt_input = gr.Text(label="Prompt")
                    negative_prompt_input = gr.Text(label="Negative prompt", value="music")
                    random_seed_input = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        seed_input = gr.Number(label="Seed", value=0, precision=0, minimum=0)
                        num_generations_input = gr.Number(
                            label="Number of generations", value=1, precision=0, minimum=1
                        )
                    with gr.Row():
                        num_steps_input = gr.Number(label="Num steps", value=25, precision=0, minimum=1)
                        cfg_strength_input = gr.Number(label="Guidance Strength", value=4.5, minimum=1)
                    generate_button = gr.Button("Generate")
                with gr.Column():
                    video_output = gr.Video(label="Generated Video")
                    
                    duration_input = gr.Number(label="Duration (sec)", value=8, minimum=1)
                    open_folder_button = gr.Button("Open outputs folder", visible=False)

            generate_button.click(
                fn=video_to_audio,
                inputs=[
                    video_input,
                    prompt_input,
                    negative_prompt_input,
                    random_seed_input,
                    seed_input,
                    num_steps_input,
                    cfg_strength_input,
                    duration_input,
                    num_generations_input,
                ],
                outputs=[video_output, seed_input, open_folder_button],
            )
            open_folder_button.click(fn=lambda: open_folder(output_dir))

            gr.Examples(
                examples=[
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_beach.mp4',
                        'waves, seagulls',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_serpent.mp4',
                        '',
                        'music',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_seahorse.mp4',
                        'bubbles',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_india.mp4',
                        'Indian holy music',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_galloping.mp4',
                        'galloping',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_kraken.mp4',
                        'waves, storm',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/mochi_storm.mp4',
                        'storm',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_spring.mp4',
                        '',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_typing.mp4',
                        'typing',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/hunyuan_wake_up.mp4',
                        '',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                    [
                        'https://huggingface.co/hkchengrex/MMAudio/resolve/main/examples/sora_nyc.mp4',
                        '',
                        '',
                        True,
                        0,
                        25,
                        4.5,
                        10,
                        1,
                    ],
                ],
                inputs=[
                    video_input,
                    prompt_input,
                    negative_prompt_input,
                    random_seed_input,
                    seed_input,
                    num_steps_input,
                    cfg_strength_input,
                    duration_input,
                    num_generations_input,
                ],
            )

        with gr.Tab("Text-to-Audio"):
            with gr.Row():
                with gr.Column():
                    prompt_input_text = gr.Text(label="Prompt")
                    negative_prompt_input_text = gr.Text(label="Negative prompt")
                    random_seed_input_text = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        seed_input_text = gr.Number(label="Seed", value=0, precision=0, minimum=0)
                        num_generations_input_text = gr.Number(
                            label="Number of generations", value=1, precision=0, minimum=1
                        )
                    with gr.Row():
                        num_steps_input_text = gr.Number(label="Num steps", value=25, precision=0, minimum=1)
                        cfg_strength_input_text = gr.Number(label="Guidance Strength", value=4.5, minimum=1)
                    generate_button_text = gr.Button("Generate")
                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio")
                    
                    duration_input_text = gr.Number(label="Duration (sec)", value=8, minimum=1)
                    open_folder_button_text = gr.Button("Open outputs folder", visible=False)

            generate_button_text.click(
                fn=text_to_audio,
                inputs=[
                    prompt_input_text,
                    negative_prompt_input_text,
                    random_seed_input_text,
                    seed_input_text,
                    num_steps_input_text,
                    cfg_strength_input_text,
                    duration_input_text,
                    num_generations_input_text,
                ],
                outputs=[audio_output, seed_input_text, open_folder_button_text],
            )
            open_folder_button_text.click(fn=lambda: open_folder(output_dir))

    demo.launch(inbrowser=True, allowed_paths=[output_dir], share=args.share)