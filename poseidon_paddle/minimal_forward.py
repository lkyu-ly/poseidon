import sys
from pathlib import Path

import paddle

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "poseidon_paddle"
MODEL_DIR = REPO_ROOT / "models" / "camlab-ethz_Poseidon-T_paddle"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from scOT.model import ScOT
from scOT.utils import resolve_runtime_device


def main() -> None:
    device = paddle.set_device(resolve_runtime_device())
    model = ScOT.from_pretrained(str(MODEL_DIR))
    model.eval()
    config = model.config
    numel = config.num_channels * config.image_size * config.image_size
    pixel_values = paddle.linspace(-1.0, 1.0, numel, dtype=paddle.float32).reshape(
        [1, config.num_channels, config.image_size, config.image_size]
    )
    time = None
    if getattr(config, "use_conditioning", False):
        time = paddle.to_tensor([0.25], dtype=paddle.float32)
    with paddle.no_grad():
        outputs = model(pixel_values=pixel_values, time=time)
    prediction = outputs.output
    print(f"model_dir: {MODEL_DIR}")
    print(f"device: {device}")
    print(f"input_shape: {tuple(pixel_values.shape)}")
    print(f"output_shape: {tuple(prediction.shape)}")
    print("output_sample:")
    print(prediction[0, 0, :4, :4])


if __name__ == "__main__":
    main()
