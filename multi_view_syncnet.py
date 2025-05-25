import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import soundfile as sf
from pathlib import Path
from shutil import rmtree
import argparse
from SyncNetInstance import SyncNetInstance
from types import SimpleNamespace

# ================================
# ARGUMENT PARSER
# ================================
parser = argparse.ArgumentParser(description="Run SyncNet on a video segment and compute AV offset.")
parser.add_argument("--video_path", type=str, required=True, help="Path to input video file (.avi)")
parser.add_argument("--tmp_dir", type=str, required=True, help="Directory to store temporary segment files")
parser.add_argument("--model_path", type=str, required=True, help="Path to SyncNet model file (.model)")
parser.add_argument("--segment_len", type=int, default=5, help="Segment length in seconds")
parser.add_argument("--stride", type=int, default=5, help="Stride in seconds")
args = parser.parse_args()

# ================================
# MAIN LOGIC
# ================================

def run_cmd(command):
    subprocess.call(command, shell=True)

def extract_audio(video_path, audio_path):
    run_cmd(f"ffmpeg -y -i {video_path} -ac 1 -ar 16000 -vn {audio_path}")

def get_duration(filepath):
    with sf.SoundFile(filepath) as f:
        return len(f) / f.samplerate

def save_segment(video_path, start_sec, duration, out_path):
    run_cmd(f"ffmpeg -y -i {video_path} -ss {start_sec} -t {duration} -c copy {out_path}")

def main():
    os.makedirs(args.tmp_dir, exist_ok=True)
    full_audio_path = os.path.join(args.tmp_dir, "full.wav")
    extract_audio(args.video_path, full_audio_path)
    audio_duration = get_duration(full_audio_path)
    video_duration = audio_duration  # Assuming AV alignment

    print(f"[INFO] Full duration: {video_duration:.2f}s")
    segment_offsets = []

    s = SyncNetInstance()
    s.loadParameters(args.model_path)
    s.eval()

    segment_id = 0
    for start_sec in np.arange(0, video_duration, args.stride):
        print(f"[DEBUG] start_sec = {start_sec}")
        if start_sec + args.segment_len > video_duration:
            print(f"[WARNING] Skipping segment at {start_sec:.2f}s due to insufficient length.")
            continue

        segment_video = os.path.join(args.tmp_dir, f"segment_{segment_id}.avi")
        segment_audio = os.path.join(args.tmp_dir, f"segment_{segment_id}.wav")

        save_segment(args.video_path, start_sec, args.segment_len, segment_video)
        save_segment(full_audio_path, start_sec, args.segment_len, segment_audio)

        print(f"[INFO] Processing segment {segment_id} ({start_sec:.2f}-{start_sec + args.segment_len:.2f}s)...")

        opt = SimpleNamespace(
            tmp_dir=args.tmp_dir,
            reference=f"segment_{segment_id}",
            batch_size=20,
            vshift=15,
            data_dir=args.tmp_dir,
            saveframes=False
        )

        offset, conf, _ = s.evaluate(opt=opt, videofile=segment_video)
        print(f"  -> Offset: {offset:.3f}, Conf: {conf:.3f}")
        segment_offsets.append((start_sec, offset, conf))

        segment_id += 1

    # Plotting
    if segment_offsets:
        times, offsets, confs = zip(*segment_offsets)

        frame_rate = 25  # assumed frame rate
        offsets_sec = [frame / frame_rate for frame in offsets]  # convert to seconds

        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_path = os.path.join(args.tmp_dir, f"{base_name}_offset_plot.png")

        plt.figure(figsize=(10, 4))
        plt.plot(times, offsets_sec, marker='o')
        plt.xlabel("Start Time (s)")
        plt.ylabel("Offset (seconds)")
        plt.title("Audio Offset Over Time")
        plt.suptitle(f"Segment Len: {args.segment_len}s, Stride: {args.stride}s", fontsize=10, y=0.94)
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()
        print(f"[INFO] Offset plot saved to: {output_path}")
    else:
        print("[WARNING] No valid segments were processed.")

if __name__ == "__main__":
    main()

