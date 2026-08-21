"""Build a watchable replay from captured simulation frames."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReviewError(ValueError):
    """The run directory has no captured camera frames to review."""


@dataclass(frozen=True, slots=True)
class ReviewArtifacts:
    html: Path
    data: Path
    videos: tuple[Path, ...]
    notes: tuple[str, ...] = ()


def write_review(run_dir: Path, *, encode: bool = False) -> ReviewArtifacts:
    """Write review.html (and optional MP4s) next to a simulation run."""
    run_dir = run_dir.resolve()
    cameras = collect_camera_frames(run_dir)
    if not cameras:
        raise ReviewError(f"no captured frames under {run_dir / 'frames'}")
    data = build_review_data(run_dir, cameras)
    data_path = run_dir / "review.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    html_path = run_dir / "review.html"
    html_path.write_text(_html(data), encoding="utf-8")
    videos: list[Path] = []
    notes: list[str] = []
    if encode:
        encoded, notes = _encode_videos(run_dir, cameras, float(data["fps"]))
        videos.extend(encoded)
    return ReviewArtifacts(html_path, data_path, tuple(videos), tuple(notes))


def collect_camera_frames(run_dir: Path) -> dict[str, list[Path]]:
    frames_dir = run_dir / "frames"
    cameras: dict[str, list[Path]] = {}
    if not frames_dir.is_dir():
        return cameras
    for child in sorted(frames_dir.iterdir()):
        if not child.is_dir():
            continue
        pngs = sorted(path for path in child.iterdir() if path.suffix.lower() == ".png")
        if pngs:
            cameras[child.name] = pngs
    return cameras


def build_review_data(run_dir: Path, cameras: dict[str, list[Path]]) -> dict[str, Any]:
    by_index: dict[int, dict[str, str]] = {}
    for camera, paths in cameras.items():
        for path in paths:
            if not path.stem.isdigit():
                continue
            index = int(path.stem)
            by_index.setdefault(index, {})[camera] = _relative(run_dir, path)
    phases = _phases_from_index(run_dir)
    frames = []
    for index in sorted(by_index):
        row: dict[str, Any] = {"i": index, "files": by_index[index]}
        if index in phases:
            row["phase"] = phases[index]
        frames.append(row)
    return {
        "title": run_dir.name,
        "fps": _fps(run_dir),
        "cameras": list(cameras),
        "frames": frames,
        "checkpoints": _checkpoints(run_dir),
    }


def _relative(run_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(run_dir).as_posix()


def _fps(run_dir: Path) -> float:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return 12.5
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 12.5
    if not isinstance(manifest, dict):
        return 12.5
    timestep = manifest.get("timestep_s")
    stride = manifest.get("capture_stride")
    if (
        isinstance(timestep, (int, float))
        and isinstance(stride, (int, float))
        and timestep > 0
        and stride > 0
    ):
        return round(1.0 / (float(timestep) * float(stride)), 4)
    return 12.5


def _phases_from_index(run_dir: Path) -> dict[int, str]:
    path = run_dir / "frames" / "index.jsonl"
    if not path.is_file():
        return {}
    phases: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        index = row.get("index")
        phase = row.get("phase")
        if isinstance(index, int) and isinstance(phase, str) and phase:
            phases[index] = phase
    return phases


def _checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "explore.json"
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = report.get("checkpoints") if isinstance(report, dict) else None
    if not isinstance(items, list):
        return []
    checkpoints: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("checkpoint_id"), str):
            continue
        checkpoints.append(
            {
                "id": item["checkpoint_id"],
                "skill": item.get("skill"),
                "ok": bool(item.get("ok")),
                "frame": item.get("start_frame"),
            }
        )
    return checkpoints


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path.home() / ".local" / "bin" / "ffmpeg",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _encode_videos(
    run_dir: Path, cameras: dict[str, list[Path]], fps: float
) -> tuple[list[Path], list[str]]:
    dest_dir = run_dir / "videos"
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    notes: list[str] = []
    ffmpeg = _find_ffmpeg()
    for camera, frames in cameras.items():
        dest = dest_dir / f"{camera}.mp4"
        if ffmpeg is not None:
            error = _encode_ffmpeg(ffmpeg, frames, dest, fps)
            if error is None and dest.is_file():
                written.append(dest)
                continue
            if error:
                notes.append(f"{camera}: ffmpeg failed ({error})")
        fallback = _encode_pillow(frames, dest_dir / camera, fps)
        if fallback is not None:
            written.append(fallback)
            if ffmpeg is None:
                notes.append(f"{camera}: wrote {fallback.name} without ffmpeg")
            continue
        if ffmpeg is None:
            notes.append(
                f"{camera}: no ffmpeg on PATH and Pillow is unavailable for a GIF/WebP fallback"
            )
    return written, notes


def _encode_ffmpeg(ffmpeg: str, frames: list[Path], dest: Path, fps: float) -> str | None:
    """Concat demuxer is portable; macOS ffmpeg often lacks -pattern_type glob."""
    duration = 1.0 / fps if fps > 0 else 0.08
    listing = dest.with_suffix(".concat.txt")
    lines = []
    for frame in frames:
        lines.append(f"file {json.dumps(str(frame.resolve()))}")
        lines.append(f"duration {duration:.6f}")
    lines.append(f"file {json.dumps(str(frames[-1].resolve()))}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-vf",
        "pad=2*ceil(iw/2):2*ceil(ih/2)",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    if completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0:
        return None
    detail = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
    return detail[:300] if detail else f"exit {completed.returncode}"


def _encode_pillow(frames: list[Path], dest_stem: Path, fps: float) -> Path | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    images = [Image.open(path).convert("RGBA") for path in frames]
    try:
        duration_ms = max(1, int(round(1000 / fps))) if fps > 0 else 80
        first, rest = images[0], images[1:]
        webp = dest_stem.with_suffix(".webp")
        try:
            first.save(
                webp,
                format="WEBP",
                save_all=True,
                append_images=rest,
                duration=duration_ms,
                loop=0,
            )
            if webp.is_file() and webp.stat().st_size > 0:
                return webp
        except (OSError, ValueError):
            pass
        gif = dest_stem.with_suffix(".gif")
        first.save(
            gif,
            format="GIF",
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        if gif.is_file() and gif.stat().st_size > 0:
            return gif
        return None
    finally:
        for image in images:
            image.close()


def _html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return _TEMPLATE.replace("%%DATA%%", payload)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>%%TITLE%%</title>
<style>
:root { color-scheme: dark; }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #111; color: #eee; }
header { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
  padding: 12px 16px; background: #1c1c1c; border-bottom: 1px solid #333; }
h1 { font-size: 16px; margin: 0; font-weight: 600; }
.meta { color: #aaa; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 8px; padding: 12px; }
.cam { background: #000; border: 1px solid #333; border-radius: 6px; overflow: hidden; }
.cam h2 { margin: 0; padding: 6px 8px; font-size: 12px; font-weight: 600;
  background: #1a1a1a; color: #ccc; }
.cam img { display: block; width: 100%; background: #000; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  padding: 0 16px 12px; }
button, select { background: #2a2a2a; color: #eee; border: 1px solid #555;
  border-radius: 4px; padding: 6px 10px; cursor: pointer; }
button:hover { background: #333; }
input[type=range] { flex: 1; min-width: 160px; }
.marks { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 16px 16px; }
.marks button { font-size: 12px; }
.marks .fail { border-color: #a44; color: #f88; }
.marks .pass { border-color: #4a6; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <span class="meta" id="status"></span>
</header>
<div class="grid" id="grid"></div>
<div class="controls">
  <button id="play" type="button">Play</button>
  <button id="prev" type="button">-1</button>
  <button id="next" type="button">+1</button>
  <input id="scrub" type="range" min="0" value="0"/>
  <select id="speed">
    <option value="0.25">0.25×</option>
    <option value="0.5">0.5×</option>
    <option value="1" selected>1×</option>
    <option value="2">2×</option>
    <option value="4">4×</option>
  </select>
</div>
<div class="marks" id="marks"></div>
<script>
const DATA = %%DATA%%;
const images = {};
const grid = document.getElementById("grid");
DATA.cameras.forEach((camera) => {
  const wrap = document.createElement("div");
  wrap.className = "cam";
  wrap.innerHTML = "<h2></h2><img alt=\\"\\"/>";
  wrap.querySelector("h2").textContent = camera;
  grid.appendChild(wrap);
  images[camera] = wrap.querySelector("img");
});
document.getElementById("title").textContent = DATA.title + " review";
const scrub = document.getElementById("scrub");
const status = document.getElementById("status");
const playBtn = document.getElementById("play");
const n = DATA.frames.length;
scrub.max = Math.max(0, n - 1);
let index = 0;
let timer = null;
function show(i) {
  if (!n) return;
  index = Math.max(0, Math.min(n - 1, i));
  scrub.value = String(index);
  const frame = DATA.frames[index];
  DATA.cameras.forEach((camera) => {
    const src = frame.files[camera];
    images[camera].src = src ? src : "";
  });
  const phase = frame.phase ? "  phase=" + frame.phase : "";
  status.textContent = "frame " + (index + 1) + "/" + n + "  #" + frame.i + phase;
}
function stop() {
  if (timer !== null) { window.clearInterval(timer); timer = null; }
  playBtn.textContent = "Play";
}
function play() {
  if (timer !== null) { stop(); return; }
  playBtn.textContent = "Pause";
  const tick = () => {
    if (index >= n - 1) { stop(); return; }
    show(index + 1);
  };
  const fps = DATA.fps || 12.5;
  const speed = parseFloat(document.getElementById("speed").value) || 1;
  timer = window.setInterval(tick, Math.max(16, 1000 / (fps * speed)));
}
playBtn.addEventListener("click", play);
document.getElementById("prev").addEventListener("click", () => { stop(); show(index - 1); });
document.getElementById("next").addEventListener("click", () => { stop(); show(index + 1); });
scrub.addEventListener("input", () => { stop(); show(parseInt(scrub.value, 10)); });
document.getElementById("speed").addEventListener("change", () => {
  if (timer !== null) { stop(); play(); }
});
document.addEventListener("keydown", (event) => {
  if (event.key === " ") { event.preventDefault(); play(); }
  if (event.key === "ArrowLeft") { stop(); show(index - 1); }
  if (event.key === "ArrowRight") { stop(); show(index + 1); }
});
const marks = document.getElementById("marks");
(DATA.checkpoints || []).forEach((cp) => {
  const button = document.createElement("button");
  button.type = "button";
  button.className = cp.ok ? "pass" : "fail";
  button.textContent = cp.id + (cp.skill ? " · " + cp.skill : "") + (cp.ok ? "" : " FAIL");
  button.addEventListener("click", () => {
    stop();
    if (typeof cp.frame === "number") {
      const pos = DATA.frames.findIndex((f) => f.i === cp.frame);
      show(pos >= 0 ? pos : 0);
    }
  });
  marks.appendChild(button);
});
show(0);
</script>
</body>
</html>
""".replace("%%TITLE%%", "Simulation review")
