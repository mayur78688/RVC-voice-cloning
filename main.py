import os
import sys
from dotenv import load_dotenv
from scipy.io import wavfile

now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config
from infer.modules.vc.modules import VC


def main():
    # Load variables from .env file
    load_dotenv()

    # Read environment variables
    f0up_key = int(os.getenv("F0UP_KEY", 0))
    input_path = os.getenv("INPUT_PATH")
    index_path = os.getenv("INDEX_PATH")
    f0method = os.getenv("F0METHOD", "harvest")
    opt_path = os.getenv("OPT_PATH")
    model_name = "assets/Synthesizer_inputs.pth"
    index_rate = float(os.getenv("INDEX_RATE", 0.66))
    device = os.getenv("DEVICE", "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu")
    is_half = os.getenv("IS_HALF", "False").lower() in ("true", "1", "yes")
    filter_radius = int(os.getenv("FILTER_RADIUS", 3))
    resample_sr = int(os.getenv("RESAMPLE_SR", 0))
    rms_mix_rate = float(os.getenv("RMS_MIX_RATE", 1))
    protect = float(os.getenv("PROTECT", 0.33))

    # Initialize config and model
    config = Config()
    config.device = device
    config.is_half = is_half

    vc = VC(config)
    vc.get_vc(model_name)

    # Run inference
    _, wav_opt = vc.vc_single(
        0,
        input_path,
        f0up_key,
        None,
        f0method,
        index_path,
        None,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )

    # Save output
    wavfile.write(opt_path, wav_opt[0], wav_opt[1])
    print(f"✅ Conversion complete. Output saved at: {opt_path}")


if __name__ == "__main__":
    main()
