import numpy as np


# 手动把两个待比较的 tensor/array 填到这里。
# 支持 numpy.ndarray、torch.Tensor、paddle.Tensor，或可被 np.asarray 转换的对象。
TENSOR_TORCH = [
    [-0.41445526, -0.44486004, -0.43125966, -0.43537739],
    [-0.47993648, -0.47820330, -0.43621975, -0.44891670],
    [-0.48770094, -0.46072891, -0.42307290, -0.44350287],
    [-0.46613464, -0.44732872, -0.41907197, -0.44356176],
]
TENSOR_PADDLE = [
    [-0.41445556, -0.44486043, -0.43125990, -0.43537751],
    [-0.47993690, -0.47820380, -0.43622011, -0.44891685],
    [-0.48770136, -0.46072933, -0.42307323, -0.44350299],
    [-0.46613494, -0.44732901, -0.41907227, -0.44356185],
]


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def main():
    a = to_numpy(TENSOR_TORCH)
    b = to_numpy(TENSOR_PADDLE)

    if a is None or b is None:
        raise ValueError("请先在脚本顶部填写 TENSOR_TORCH 和 TENSOR_PADDLE。")

    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")

    diff = a - b
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(a), 1e-12)
    rel_diff = abs_diff / denom

    print(f"shape: {a.shape}")
    print(f"dtype: {a.dtype} vs {b.dtype}")
    print(f"max_abs_diff: {abs_diff.max():.12f}")
    print(f"mean_abs_diff: {abs_diff.mean():.12f}")
    print(f"rmse: {np.sqrt(np.mean(diff ** 2)):.12f}")
    print(f"max_rel_diff: {rel_diff.max():.12f}")
    print(f"mean_rel_diff: {rel_diff.mean():.12f}")
    print(f"allclose(atol=1e-6, rtol=1e-6): {np.allclose(a, b, atol=1e-6, rtol=1e-6)}")
    print(f"allclose(atol=1e-7, rtol=1e-7): {np.allclose(a, b, atol=1e-7, rtol=1e-7)}")


if __name__ == "__main__":
    main()
