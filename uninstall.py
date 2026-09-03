#!/usr/bin/env python3
"""
llama-deploy 卸载工具
安全地清理所有已部署的文件

用法：python uninstall.py
"""

# Windows UTF-8 修复
import sys
import io
import os

if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul 2>&1")
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

import platform
import shutil
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
IS_WIN = platform.system() == "Windows"


def get_size_str(path: Path) -> str:
    """计算目录/文件大小"""
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        return "0 B"
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.0f} MB"
    elif size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def stop_server():
    """停止运行中的服务器"""
    print("🔍 检查运行中的服务器...")

    # 按 PID 文件停止
    pid_file = BASE_DIR / ".llama-server.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if IS_WIN:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, 15)
            print(f"   ✅ 已停止服务器 (PID: {pid})")
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)

    # 按进程名停止
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    except Exception:
        pass


def scan_files() -> dict:
    """扫描所有可清理的文件"""
    items = {}

    paths = {
        "llama.cpp（引擎）":   BASE_DIR / "llama.cpp",
        "models（模型文件）":   BASE_DIR / "models",
        ".hf-venv（下载工具）": BASE_DIR / ".hf-venv",
    }
    for name, path in paths.items():
        if path.exists():
            items[name] = {"path": path, "size": get_size_str(path), "type": "dir"}

    temp_files = [
        ".llama-server.pid", ".llama-server.log", ".deploy.log",
        "config.jsonc",
    ]
    for fname in temp_files:
        p = BASE_DIR / fname
        if p.exists():
            items[f"{fname}"] = {"path": p, "size": get_size_str(p), "type": "file"}

    return items


def main():
    print("""
╔══════════════════════════════════════════════╗
║  🗑️  llama-deploy 卸载工具                   ║
╚══════════════════════════════════════════════╝
    """)
    print(f"📁 项目目录: {BASE_DIR}\n")

    # 停止服务器
    stop_server()

    # 扫描文件
    items = scan_files()

    if not items:
        print("✅ 没有需要清理的文件")
        return

    # 显示清单
    print("📋 以下文件/目录将被删除:\n")
    total_display = []
    for name, info in items.items():
        icon = "📁" if info["type"] == "dir" else "📄"
        print(f"   {icon} {name:<30s} {info['size']:>10s}    {info['path']}")
        total_display.append(name)

    print(f"\n   共 {len(items)} 项")

    # 选择模式
    print("""
请选择卸载模式:
  [1] 完全卸载 — 删除所有文件（引擎 + 模型 + 配置）
  [2] 仅删除引擎 — 保留模型文件和配置
  [3] 仅删除模型 — 保留引擎和配置
  [4] 仅清理临时文件 — 保留引擎、模型、配置
  [0] 取消
    """)

    choice = input("请输入选项 [0-4]: ").strip()

    if choice == "0":
        print("\n⏹️  已取消")
        return

    # 确定删除范围
    to_delete = {}

    if choice == "1":
        to_delete = items
    elif choice == "2":
        for name, info in items.items():
            if "llama.cpp" in name or ".hf-venv" in name:
                to_delete[name] = info
    elif choice == "3":
        for name, info in items.items():
            if "models" in name:
                to_delete[name] = info
    elif choice == "4":
        for name, info in items.items():
            if info["type"] == "file" and "config" not in name:
                to_delete[name] = info
    else:
        print("❌ 无效选项")
        return

    if not to_delete:
        print("\n✅ 没有需要删除的项目")
        return

    # 二次确认
    print(f"\n⚠️  即将删除 {len(to_delete)} 项:")
    for name in to_delete:
        print(f"   • {name}")

    confirm = input("\n确认删除？输入 yes 继续: ").strip().lower()
    if confirm != "yes":
        print("\n⏹️  已取消")
        return

    # 执行删除
    print("\n🗑️  正在删除...\n")
    success = 0
    failed = 0

    for name, info in to_delete.items():
        try:
            if info["type"] == "dir":
                shutil.rmtree(info["path"])
            else:
                info["path"].unlink()
            print(f"   ✅ 已删除: {name}")
            success += 1
        except Exception as e:
            print(f"   ❌ 删除失败: {name} — {e}")
            failed += 1

    print(f"\n{'─' * 50}")
    print(f"✅ 成功删除: {success} 项")
    if failed:
        print(f"❌ 删除失败: {failed} 项")

    # 提示
    remaining = [
        "deploy.py", "run.py", "manager.py", "uninstall.py", "README.md"
    ]
    remaining_exists = [f for f in remaining if (BASE_DIR / f).exists()]

    if remaining_exists:
        print(f"\n💡 以下脚本文件未删除（需手动删除整个目录）:")
        for f in remaining_exists:
            print(f"   📄 {f}")
        print(f"\n   完全删除项目:")
        if IS_WIN:
            print(f'   rmdir /s /q "{BASE_DIR}"')
        else:
            print(f'   rm -rf "{BASE_DIR}"')

    print("\n👋 卸载完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户取消")
