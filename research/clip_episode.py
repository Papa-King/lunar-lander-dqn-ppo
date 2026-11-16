"""Extract an exact complete episode from a ten-episode recording by frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio

from metrics import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    record = json.loads(source.with_suffix(".json").read_text())
    if not 1 <= args.episode <= len(record["episodes"]):
        parser.error("Episode number is outside the recorded range")
    start = sum(item["steps"] + 1 for item in record["episodes"][:args.episode - 1])
    episode = record["episodes"][args.episode - 1]
    frame_count = episode["steps"] + 1
    end = start + frame_count
    fps = int(record["fps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.mp4")
    written = 0
    reader = imageio.get_reader(source)
    try:
        with imageio.get_writer(temporary, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for index, frame in enumerate(reader):
                if index >= end:
                    break
                if index >= start:
                    writer.append_data(frame)
                    written += 1
    finally:
        reader.close()
    if written != frame_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {frame_count} frames, wrote {written}")
    temporary.replace(output)
    model_path = Path(record["model"])
    write_json(output.with_suffix(".json"), {
        "source_video": str(source), "source_episode": args.episode,
        "source_start_frame": start, "source_end_frame_exclusive": end,
        "frames": written, "fps": fps, "duration_seconds": written / fps,
        "reward": episode["reward"], "outcome": episode["outcome"],
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "video": str(output),
    })
    print(f"Saved {output} ({written / fps:.2f}s, episode {args.episode})")


if __name__ == "__main__":
    main()
