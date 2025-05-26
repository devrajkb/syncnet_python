
import os
import json
import random
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import soundfile as sf
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def shift_audio(audio, sr, shift_sec):
    shift_samples = int(sr * shift_sec)
    if shift_samples > 0:
        shifted = np.concatenate((np.zeros(shift_samples), audio[:-shift_samples]))
    elif shift_samples < 0:
        shifted = np.concatenate((audio[-shift_samples:], np.zeros(-shift_samples)))
    else:
        shifted = audio
    return shifted

def shift_audio_segmented(audio, sr, duration, interval_sec, shift_range):
    segments = []
    offset_log = []
    total_segments = int(np.ceil(duration / interval_sec))

    for i in range(total_segments):
        start = int(i * interval_sec * sr)
        end = int(min((i + 1) * interval_sec * sr, len(audio)))
        segment = audio[start:end]
        shift = round(random.uniform(*shift_range), 3)
        shifted = shift_audio(segment, sr, shift)
        offset_log.append({
            "start_sec": round(i * interval_sec, 2),
            "end_sec": round(min((i + 1) * interval_sec, duration), 2),
            "offset_sec": shift
        })
        segments.append(shifted)

    shifted_audio = np.concatenate(segments)
    return shifted_audio, offset_log

def plot_offset(offset_log, output_path):
    times = [entry["start_sec"] for entry in offset_log]
    values = [entry["offset_sec"] for entry in offset_log]

    plt.figure(figsize=(10, 4))
    plt.plot(times, values, marker='o')
    plt.title("Audio Offset Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Offset (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_offset_csv(offset_log, csv_path):
    df = pd.DataFrame(offset_log)
    df.to_csv(csv_path, index=False)

def get_color_for_offset(offset):
    abs_offset = abs(offset)
    if abs_offset < 0.1:
        return (0, 255, 0)
    elif abs_offset < 0.2:
        return (255, 165, 0)
    else:
        return (255, 0, 0)

def save_frame_overlay_debug(video, offset_log, out_debug_path, frame_offset_csv_path):
    fps = video.fps
    interval_frames = int(offset_log[0]["end_sec"] * fps)
    frames = list(video.iter_frames(fps=fps))

    debug_writer = FFMPEG_VideoWriter(out_debug_path, size=video.size, fps=fps, codec="libx264")

    frame_records = []
    for i, frame in enumerate(frames):
        segment_idx = min(i // interval_frames, len(offset_log) - 1)
        offset_sec = offset_log[segment_idx]["offset_sec"]
        offset_frames = int(round(offset_sec * fps))
        color = get_color_for_offset(offset_sec)

        # Draw text overlay
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        text = f"Offset: {offset_sec:+.2f}s ({offset_frames:+d} frames)"
        draw.rectangle([10, 10, 310, 50], fill=(0, 0, 0, 128))
        draw.text((20, 20), text, fill=color)
        debug_writer.write_frame(np.array(img))

        frame_records.append({
            "frame_index": i,
            "time_sec": round(i / fps, 3),
            "offset_sec": offset_sec,
            "offset_frames": offset_frames
        })

    debug_writer.close()
    pd.DataFrame(frame_records).to_csv(frame_offset_csv_path, index=False)

def process_single_video(video_file, output_dir, interval_sec, shift_range):
    video_path = Path(video_file)
    video_name = video_path.stem
    input_ext = video_path.suffix.lower()

    output_video_path = os.path.join(output_dir, f"{video_name}_shifted{input_ext}")
    offset_log_path = os.path.join(output_dir, f"{video_name}.offset_log.json")
    offset_csv_path = os.path.join(output_dir, f"{video_name}.offset_log.csv")
    offset_plot_path = os.path.join(output_dir, f"{video_name}.offset_plot.png")
    debug_overlay_path = os.path.join(output_dir, f"{video_name}_debug_overlay.mp4")
    frame_offset_csv_path = os.path.join(output_dir, f"{video_name}_frame_offset.csv")

    video = VideoFileClip(str(video_path))
    duration = video.duration
    sr = 16000

    temp_audio_path = os.path.join(output_dir, f"{video_name}_temp.wav")
    video.audio.write_audiofile(temp_audio_path, fps=sr)
    audio_array, _ = sf.read(temp_audio_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    shifted_audio, offset_log = shift_audio_segmented(
        audio_array, sr, duration, interval_sec, shift_range
    )

    shifted_audio_path = os.path.join(output_dir, f"{video_name}_shifted.wav")
    sf.write(shifted_audio_path, shifted_audio, sr)

    with open(offset_log_path, 'w') as f:
        json.dump(offset_log, f, indent=4)

    save_offset_csv(offset_log, offset_csv_path)
    plot_offset(offset_log, offset_plot_path)
    save_frame_overlay_debug(video, offset_log, debug_overlay_path, frame_offset_csv_path)

    new_audio = AudioFileClip(shifted_audio_path)
    new_video = video.with_audio(new_audio)

    if input_ext == '.avi':
        new_video.write_videofile(output_video_path, codec='mpeg4', audio_codec='libmp3lame')
    else:
        new_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    os.remove(temp_audio_path)
    os.remove(shifted_audio_path)

    print("[OK] Saved shifted video:", output_video_path)
    print("[INFO] Offset JSON:", offset_log_path)
    print("[PLOT] Offset Plot:", offset_plot_path)
    print("[CSV] Offset Table:", offset_csv_path)
    print("[CSV] Frame-Level Offset Table:", frame_offset_csv_path)
    print("[DEBUG] Debug Overlay Video:", debug_overlay_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file", required=True, help="Single video file path (.mp4 or .avi)")
    parser.add_argument("--output_dir", required=True, help="Where to save shifted video and logs")
    parser.add_argument("--interval_sec", type=float, default=2.0, help="Interval in seconds for changing offset")
    parser.add_argument("--min_shift", type=float, default=-0.3, help="Min audio shift in seconds")
    parser.add_argument("--max_shift", type=float, default=0.3, help="Max audio shift in seconds")
    args = parser.parse_args()

    shift_range = (args.min_shift, args.max_shift)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    process_single_video(args.video_file, args.output_dir, args.interval_sec, shift_range)
