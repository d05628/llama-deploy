"""固定的 agent 测试题组。每道题：提示词、需要的命令、工作目录准备、自动评分。

题目刻意覆盖不同能力：
    movie   找文件 + 多媒体理解（抽帧看图 + 语音转写）   用户出题
    blender 调用外部工具生成 3D 动画、报错后自行修正       用户出题
    svg     纯创意代码生成（鹈鹕骑自行车，业内常用的 SVG 题）
    bugfix  在现成的多文件代码库里定位根因、最小修改、自行验证；
            受"不许改测试"约束，完全可自动客观评分 —— 弥补前几题难以客观打分、
            也没考到"读懂已有代码并按规范修改"这一日常最常见的 agent 工作
"""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
RETRY = "- 如果某条命令被拒绝或失败，读懂原因后换一种写法或换一个工具，不要原样重复同一条命令。\n"
VIDEO_EXT = (".mp4", ".mkv", ".avi", ".rmvb", ".mov", ".wmv", ".flv", ".ts", ".m4v")


def _data_drives():
    """C 盘以外的本地盘符，如 ["D:/", "E:/"]。"""
    import string
    import os
    return [f"{d}:/" for d in string.ascii_uppercase if d not in "ABC" and os.path.isdir(f"{d}:/")]


def _ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _frame_md5(path: Path, second: float) -> str:
    try:
        out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(second), "-i", str(path), "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=60)
        return hashlib.md5(out.stdout).hexdigest() if out.stdout else ""
    except Exception:
        return ""


def _save_frame(video: Path, second: float, dest: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", str(video), "-frames:v", "1", str(dest)],
                   capture_output=True, timeout=60)


# ---------------------------------------------------------------- movie
def movie_setup(work: Path):
    pass


def movie_check(work: Path) -> dict:
    report = next((p for p in work.rglob("*.md") if "报告" in p.name), None) or next(iter(work.rglob("*.md")), None)
    text = report.read_text(encoding="utf-8", errors="replace") if report else ""
    paths = re.findall(r"[A-Za-z]:[\\/][^\s`\"'|*<>]+?(?:%s)" % "|".join(re.escape(e) for e in VIDEO_EXT), text, re.I)
    real = [p for p in paths if Path(p).exists()]
    frames = [p for p in work.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    checks = {
        "写出报告": bool(report) and len(text) >= 200,
        "报告里的视频路径真实存在": bool(real),
        "抽取了至少 3 帧": len(frames) >= 3,
        "有按时间的画面描述": bool(re.search(r"\d{1,2}:\d{2}|\d+\s*秒|\d+s\b", text)),
    }
    return {"checks": checks, "artifacts": {"report": str(report) if report else "", "video": real[0] if real else "",
                                            "frames": [str(p) for p in sorted(frames)[:6]]}}


# ---------------------------------------------------------------- blender
def blender_setup(work: Path):
    pass


def blender_check(work: Path) -> dict:
    video = next(iter(work.rglob("stickman.mp4")), None) or next(iter(work.rglob("*.mp4")), None)
    duration = _ffprobe_duration(video) if video else 0.0
    moving = bool(video) and _frame_md5(video, 2) != _frame_md5(video, 20) and _frame_md5(video, 2) != ""
    preview = work / "_bench_preview.png"
    if video:
        _save_frame(video, 10, preview)
    checks = {
        "生成了视频": bool(video),
        "时长约 30 秒（28–32）": 28 <= duration <= 32,
        "画面在动（2 秒与 20 秒的帧不同）": moving,
        "留下了 Blender 脚本": any(work.rglob("*.py")),
    }
    return {"checks": checks, "artifacts": {"video": str(video) if video else "", "duration": round(duration, 1),
                                            "preview": str(preview) if preview.exists() else ""}}


# ---------------------------------------------------------------- svg
def svg_setup(work: Path):
    pass


def svg_check(work: Path) -> dict:
    svg = next(iter(work.rglob("pelican.svg")), None) or next(iter(work.rglob("*.svg")), None)
    valid, elements = False, 0
    if svg:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(svg).getroot()
            valid = root.tag.endswith("svg")
            elements = sum(1 for _ in root.iter())
        except Exception:
            valid = False
    return {"checks": {"生成了 SVG": bool(svg), "是合法的 SVG 文档": valid, "有一定复杂度（≥10 个元素）": elements >= 10},
            "artifacts": {"svg": str(svg) if svg else "", "elements": elements}}


# ---------------------------------------------------------------- bugfix
def _tests_hash(work: Path) -> str:
    h = hashlib.md5()
    for p in sorted((work / "tests").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def bugfix_setup(work: Path):
    shutil.copytree(FIXTURES / "invoice", work, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (work / ".bench_tests_hash").write_text(_tests_hash(work), encoding="utf-8")


def bugfix_check(work: Path) -> dict:
    run = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=work,
                         capture_output=True, text=True, timeout=120)
    summary = (run.stdout.strip().splitlines() or [""])[-1]
    untouched = (work / ".bench_tests_hash").read_text(encoding="utf-8") == _tests_hash(work)
    notes = work / "FIX_NOTES.md"
    return {"checks": {"8 个测试全部通过": run.returncode == 0 and "8 passed" in summary,
                       "没有修改测试文件": untouched,
                       "写了修复说明 FIX_NOTES.md": notes.exists() and notes.stat().st_size > 50},
            "artifacts": {"pytest": summary, "notes": str(notes) if notes.exists() else ""}}


TASKS = {
    "movie": {
        "title": "电影文件夹：解读视频开头一分钟",
        "commands": ["ffmpeg", "ffprobe", "whisper"],
        "read_dirs": _data_drives,
        "vision": True,
        "timeout": 45 * 60,
        "setup": movie_setup, "check": movie_check,
        "prompt": (
            "任务：扫描 C 盘以外的所有硬盘，找到名字里带「电影」的文件夹，从中任选一个视频，"
            "解读它开头一分钟的内容（画面讲了什么、人物/场景、有没有对白及对白内容）。\n"
            "要求与环境：\n"
            "- 只读：绝对不要修改、移动、删除任何已有文件；所有中间文件和结果都写到当前工作目录。\n"
            "- 找文件夹请先用你自带的列目录 / glob 类文件工具看各盘根目录，再往下找。\n"
            "- shell 里只允许 ffmpeg、ffprobe、whisper 三个命令，其它命令会被拒绝。\n"
            "- 用 ffmpeg 在开头一分钟里每隔约 10 秒抽一帧，缩放到宽 448 像素存成 jpg，再用你的读取文件工具逐张查看图片。\n"
            "- 用 whisper 转写开头一分钟的对白（显存已占满，必须加 --device cpu，建议 --model small）。\n"
            "- 最后写当前目录下的「报告.md」：视频完整路径、时长、分辨率、按时间的画面描述、对白摘要、一句话总结。\n" + RETRY
        ),
    },
    "blender": {
        "title": "Blender：火柴人动画，导出 30 秒视频",
        "commands": ["blender", "ffprobe", "ffmpeg"],
        "read_dirs": lambda: [],
        "vision": True,
        "timeout": 35 * 60,
        "setup": blender_setup, "check": blender_check,
        "prompt": (
            "任务：用 Blender 做一个火柴人模型，让它做简单的动作（例如挥手、走路或跳跃），导出一段 30 秒的视频。\n"
            "要求与环境：\n"
            "- Blender 已在 PATH 中（命令 blender），用后台模式运行：blender -b --python 脚本.py。\n"
            "- 在当前工作目录写 Python 脚本（bpy）：用圆柱/球体搭火柴人，加相机和灯光，用关键帧做动作，"
            "24 fps × 30 秒 = 720 帧，渲染引擎用 BLENDER_WORKBENCH（最快），分辨率 640x360，输出 MP4（FFMPEG，H264），"
            "文件名 stickman.mp4。\n"
            "- shell 里只允许 blender、ffprobe、ffmpeg 三个命令。脚本报错就读错误信息修改后重跑，直到生成 stickman.mp4。\n"
            "- 完成后用 ffprobe 确认时长约 30 秒，并用 ffmpeg 抽一帧、用你的读取文件工具看一眼，确认画面里有火柴人。\n" + RETRY
        ),
    },
    "svg": {
        "title": "SVG：鹈鹕骑自行车",
        "commands": [],
        "read_dirs": lambda: [],
        "vision": False,
        "timeout": 15 * 60,
        "setup": svg_setup, "check": svg_check,
        "prompt": (
            "Generate an SVG of a pelican riding a bicycle.\n"
            "把结果保存为当前目录下的 pelican.svg（纯 SVG，不引用任何外部资源）。"
        ),
    },
    "bugfix": {
        "title": "修 bug：让现有代码库的测试全部通过",
        "commands": ["python -m pytest"],
        "read_dirs": lambda: [],
        "vision": False,
        "timeout": 25 * 60,
        "setup": bugfix_setup, "check": bugfix_check,
        "prompt": (
            "当前目录是一个小型发票计算库（invoice 包 + tests 测试）。运行 `python -m pytest -q` 会有测试失败。\n"
            "请找出并修复代码里的 bug，让全部测试通过。要求：\n"
            "- 不许修改 tests/ 目录下的任何文件，测试就是规格说明；\n"
            "- 只改真正有问题的地方，改动尽量小；\n"
            "- 自己运行 `python -m pytest -q` 确认全部通过；\n"
            "- 最后写 FIX_NOTES.md，逐条说明每个 bug 的根因和你的修改。\n"
            "- shell 里只允许 `python -m pytest` 这一个命令。\n" + RETRY
        ),
    },
}
