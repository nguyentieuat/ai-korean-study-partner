import subprocess
from pathlib import Path
import shutil
import tempfile

def run_mfa_align_single(wav_path: Path, lab_path: Path, output_dir: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_corpus = Path(tmpdir)
        
        # Copy .wav và .lab vào temp folder
        shutil.copy(wav_path, tmp_corpus / wav_path.name)
        shutil.copy(lab_path, tmp_corpus / lab_path.name)

        result = subprocess.run([
            "mfa", "align",
            str(tmp_corpus),
            "korean_mfa.dict",
            "korean_mfa.zip",
            str(output_dir),
            "--clean",
            "--overwrite"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ MFA Align lỗi:", result.stderr)
            raise RuntimeError("MFA align failed.")
        else:
            print("✅ MFA Align thành công:", result.stdout)
