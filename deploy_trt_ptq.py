
# deploy_trt_ptq.py
# Build a real INT8 TensorRT engine from a PyTorch model with minimal changes.
# H100 friendly. Uses Torch-TensorRT's PTQ calibrator.
import torch
import torch_tensorrt
from torch_tensorrt import ptq
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import transforms
from typing import Tuple, Optional

def make_calib_loader(imagenet_dir: str,
                      n_imgs: Optional[int] = 512,
                      bs: int = 32,
                      size: int = 224,
                      num_workers: int = 4) -> DataLoader:
    """Create a small ImageNet loader for PTQ calibration."""
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = tv.datasets.ImageFolder(imagenet_dir, transform=tfm)
    if (n_imgs is not None) and (len(ds.samples) > n_imgs):
        ds.samples = ds.samples[:n_imgs]
        # Some torchvision versions store targets separately; guard it.
        if hasattr(ds, "targets") and len(ds.targets) > n_imgs:
            ds.targets = ds.targets[:n_imgs]
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

def _parse_precisions(spec: str):
    mapping = {
        "fp16": torch.half,
        "half": torch.half,
        "float16": torch.half,
        "int8": torch.int8,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
    }
    out = set()
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in mapping:
            out.add(mapping[tok])
    if not out:
        out = {torch.half, torch.int8}
    return out

def build_trt_int8(model: torch.nn.Module,
                   calib_loader: DataLoader,
                   input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                   cache: str = "calib.cache",
                   out_ts: str = "vit_int8.ts",
                   precisions: str = "fp16,int8"):
    """Compile a TorchScript TensorRT engine with INT8 PTQ.
    - model: eval() and to("cuda") will be applied here.
    - calib_loader: small representative data.
    - input_shape: NCHW shape used for compilation (e.g., 1,3,224,224).
    - cache: calibration cache file.
    - out_ts: output TorchScript (.ts) file of TRT engine.
    - precisions: comma-separated, e.g., 'fp16,int8' (default).
    """
    model = model.eval().to("cuda")
    enabled_precisions = _parse_precisions(precisions)

    calibrator = ptq.DataLoaderCalibrator(
        calib_loader,
        cache_file=cache,
        use_cache=False,
        algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )

    trt_mod = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(input_shape)],
        enabled_precisions=enabled_precisions,  # {torch.half, torch.int8} by default
        calibrator=calibrator,
        require_full_compilation=False,         # allow fallback to Torch for unsupported ops
        device={"device_type": torch_tensorrt.DeviceType.GPU, "gpu_id": 0},
    )
    torch.jit.save(trt_mod, out_ts)
    return out_ts
