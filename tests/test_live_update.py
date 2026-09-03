import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import deploy


@unittest.skipUnless(
    os.environ.get("RUN_LLAMA_UPDATE_INTEGRATION") == "1",
    "set RUN_LLAMA_UPDATE_INTEGRATION=1 to download and verify a real release",
)
class LiveLlamaUpdateTests(unittest.TestCase):
    def test_latest_windows_cpu_release_runs(self):
        original_base = deploy.BASE_DIR
        original_llama = deploy.LLAMA_DIR
        try:
            with tempfile.TemporaryDirectory(prefix="llama-deploy-integration-") as temp_dir:
                root = Path(temp_dir)
                deploy.BASE_DIR = root
                deploy.LLAMA_DIR = root / "llama.cpp"

                instance = object.__new__(deploy.Deployer)
                instance.upgrade_llama = False
                instance.sys = SimpleNamespace(
                    is_windows=True, is_arm=False, arch="AMD64"
                )
                instance.server_bin = deploy.LLAMA_DIR / "llama-server.exe"
                instance.cli_bin = deploy.LLAMA_DIR / "llama-cli.exe"
                instance.actual_backend = "cpu"
                instance.gpu_info = {"cuda_version": ""}
                instance.p = deploy.Printer("zh")
                instance.dl = deploy.Downloader(
                    {
                        "download": {
                            "github_mirror": "",
                            "timeout": 300,
                            "retries": 2,
                        }
                    },
                    instance.p,
                )
                instance.target_llama_tag = ""

                instance._download_llama_windows()
                for binary in (instance.server_bin, instance.cli_bin):
                    result = subprocess.run(
                        [str(binary), "--version"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=str(binary.parent),
                    )
                    print(f"LIVE_VERSION {binary.name}: {(result.stdout + result.stderr).strip()}")
                instance._verify_llama_only()

                self.assertRegex(instance.target_llama_tag, r"^b\d+$")
        finally:
            deploy.BASE_DIR = original_base
            deploy.LLAMA_DIR = original_llama


if __name__ == "__main__":
    unittest.main()
