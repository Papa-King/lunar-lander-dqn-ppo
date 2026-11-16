"""Smoothly retime the first ranked episode to ~11 s; mark it as edited."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg

from metrics import write_json


RESEARCH = Path(__file__).resolve().parent
SOURCE = RESEARCH / "videos" / "top20_from_400_slow_ppo_v3_seed42.mp4"
SOURCE_JSON = SOURCE.with_suffix(".json")
OUTPUT = RESEARCH / "videos" / "rank1_landing_edited_11s.mp4"
TARGET_SECONDS = 11.0


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_meta = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    first = source_meta["filmed"][0]
    if first["rank"] != 1:
        raise ValueError("Source video must begin with rank 1")
    fps = source_meta["fps"]
    source_frames = first["steps"] + 1
    source_seconds = source_frames / fps
    target_frames = int(TARGET_SECONDS * fps)
    stretch = TARGET_SECONDS / source_seconds
    font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if not font.is_file():
        raise FileNotFoundError(font)
    output_tmp = OUTPUT.with_name(OUTPUT.stem + ".tmp.mp4")
    if output_tmp.exists():
        raise FileExistsError(output_tmp)
    filters = (
        f"trim=end_frame={source_frames},"
        f"setpts={stretch:.12f}*(PTS-STARTPTS),"
        "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
        "tpad=stop_mode=clone:stop_duration=0.2,"
        f"trim=end_frame={target_frames},"
        f"drawtext=fontfile={font}:text='Edited playback - 11s':"
        "x=w-tw-12:y=h-th-12:fontsize=13:fontcolor=white:"
        "box=1:boxcolor=black@0.65:boxborderw=4"
    )
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-i", str(SOURCE), "-vf", filters, "-an", "-fps_mode", "cfr",
               "-r", str(fps), "-frames:v", str(target_frames),
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-y", str(output_tmp)]
    try:
        subprocess.run(command, check=True)
        reader = imageio.get_reader(output_tmp)
        try:
            actual = reader.get_meta_data()
            reader.get_data(0)
            reader.get_data(target_frames - 1)
        finally:
            reader.close()
        if abs(actual["duration"] - TARGET_SECONDS) > 0.05:
            raise RuntimeError(f"Unexpected video duration: {actual['duration']}")
        output_tmp.replace(OUTPUT)
    finally:
        output_tmp.unlink(missing_ok=True)
    write_json(OUTPUT.with_suffix(".json"), {
        "source_video": str(SOURCE), "source_rank": first["rank"],
        "source_round": first["round"], "source_test_episode": first["test_episode"],
        "source_seed": first["seed"], "source_frames": source_frames,
        "source_duration_seconds": source_seconds,
        "edited_duration_seconds": TARGET_SECONDS,
        "playback_speed_factor": source_seconds / TARGET_SECONDS,
        "technique": "uniform smooth optical-flow retiming; labelled as edited playback",
        "model_retrained": False, "environment_replayed": False,
    })
    print(f"Saved {OUTPUT}: {source_seconds:.2f}s -> {TARGET_SECONDS:.2f}s, "
          f"playback {source_seconds / TARGET_SECONDS:.3f}x")


if __name__ == "__main__":
    main()
