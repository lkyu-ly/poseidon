from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "poseidon"
MODEL_DIR = REPO_ROOT / "models" / "camlab-ethz_Poseidon-T"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scOT.model import ScOT  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(precision=8, sci_mode=False)

    model = ScOT.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    config = model.config
    numel = config.num_channels * config.image_size * config.image_size
    pixel_values = torch.linspace(
        -1.0,
        1.0,
        steps=numel,
        dtype=torch.float32,
        device=device,
    ).reshape(
        1,
        config.num_channels,
        config.image_size,
        config.image_size,
    )

    time = None
    if getattr(config, "use_conditioning", False):
        time = torch.tensor([0.25], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, time=time)

    prediction = outputs.output
    print(f"model_dir: {MODEL_DIR}")
    print(f"device: {device}")
    print(f"input_shape: {tuple(pixel_values.shape)}")
    print(f"output_shape: {tuple(prediction.shape)}")
    print("output_sample:")
    print(prediction[0, 0, :4, :4].cpu())


if __name__ == "__main__":
    main()

"""
model_dir: /home/lkyu/baidu/poseidon/models/camlab-ethz_Poseidon-T
device: cuda
input_shape: (1, 4, 128, 128)
output_shape: (1, 4, 128, 128)
output_sample:
tensor([[-0.4145, -0.4449, -0.4313, -0.4354],
        [-0.4799, -0.4782, -0.4362, -0.4489],
        [-0.4877, -0.4607, -0.4231, -0.4435],
        [-0.4661, -0.4473, -0.4191, -0.4436]])
"""
