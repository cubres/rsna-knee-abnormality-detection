# SPDX-License-Identifier: Apache-2.0
"""Original r224 arm for the Bee V15 blend. Inference only; no network; no file writes.

Recipe reproduced from nartaa's public efficiency inference cell (Apache-2.0):
https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop?scriptVersionId=356179962
(inference cell SHA-256 e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c).
The learned weights come from nartaa's public dataset V1 under separate research and
educational terms that include this competition. They are attached, never rehosted.
This module grants no rights to OAI source data.

Transform, as in the publisher's gpu_windows (CROP_FRAC 0.8, res 224, imagenet): uint8/255
float32 values from the shared 384 volume, centre crop 307 px, bilinear resize to 224
(align_corners=False), then ImageNet normalisation. One plain view. The publisher encodes all
94 windows in one call under fp16 autocast. Failed attempts fall back to chunks of 16, then to
fp32 chunks of 16, and the path used is recorded per study.

Importing this file performs no model load, file read or network request.
"""
from __future__ import annotations

import hashlib

PUBLIC_REF = "nartaa/rsna-knee-0945-efficient-224crop"
PUBLIC_VERSION = 2
PUBLIC_SESSION_ID = 356179962
PUBLIC_INFERENCE_CELL_SHA256 = "e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c"
DATASET_DIR = "rsna-knee-publication-swa-weights-20261007"
DATASET_REF = "nartaa/" + DATASET_DIR
DATASET_VERSION = 1
CHECKPOINT_NAME = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"
CHECKPOINT_BYTES = 292842755
CHECKPOINT_SHA256 = "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"
ARCH = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"
RESOLUTION = 224
CORPUS_RES = 384
CROP_FRACTION = 0.8
WINDOW_COUNT = 94
SLICE_COUNT = 96
LABEL_COUNT = 12
CHUNK = 16
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
PROBABILITY_DEFAULT = 0.5
R384_WEIGHT = 0.75
R224_WEIGHT = 0.25
# (attempt name, encoder chunk or None for all 94 at once, fp16 autocast)
ATTEMPTS = (("fp16_all94", None, True), ("fp16_chunk16", CHUNK, True), ("fp32_chunk16", CHUNK, False))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checkpoint_path(input_root):
    candidates = (input_root / DATASET_DIR / CHECKPOINT_NAME,
                  input_root / "datasets" / DATASET_REF / CHECKPOINT_NAME)
    available = [path for path in candidates if path.is_file()]
    require(bool(available), "R224 checkpoint is absent from the attached public V1 dataset")
    return available[0]


def verify_checkpoint(path):
    """Full-file SHA-256 and exact size before any deserialisation. Returns (matches, digest, size)."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return bool(digest == CHECKPOINT_SHA256 and size == CHECKPOINT_BYTES), digest, size


def make_classifier(backbone, torch):
    """Publisher head (RaptorClassifier): LayerNorm, attention MLP, softmax over windows, einsum pool."""
    nn = torch.nn

    class R224Classifier(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(backbone.num_features)
            self.att = nn.Sequential(nn.Linear(backbone.num_features, 256), nn.Tanh(),
                                     nn.Dropout(0.2), nn.Linear(256, LABEL_COUNT))
            self.clsW = nn.Parameter(torch.zeros(LABEL_COUNT, backbone.num_features))
            self.clsb = nn.Parameter(torch.zeros(LABEL_COUNT))
            self.n = LABEL_COUNT

        def forward(self, windows, chunk=None):
            batch, count = windows.shape[:2]
            flat = windows.flatten(0, 1)
            if chunk is None:
                encoded = self.backbone(flat)
            else:
                encoded = torch.cat([self.backbone(part) for part in flat.split(chunk)], dim=0)
            features = self.norm(encoded.view(batch, count, -1))
            attention = torch.softmax(self.att(features), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", attention, features)
            return (pooled * self.clsW).sum(-1) + self.clsb

    return R224Classifier(backbone)


def create_backbone(timm):
    """Publisher build_backbone for this arch: no img_size override (default 224), avg pool."""
    return timm.create_model(ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")


def load_model(path, device, timm, torch):
    """Tensor-only load (weights_only=True), strict state dict. Architecture and resolution checked."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and "model" in state, "Unsupported R224 checkpoint container")
    require(state.get("arch", ARCH) == ARCH, "R224 checkpoint architecture differs")
    require(int(state.get("res", RESOLUTION)) == RESOLUTION, "R224 checkpoint resolution differs")
    model = make_classifier(create_backbone(timm), torch)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval().to(device)
    require(all(not module.training for module in model.modules()), "R224 model has a training module")
    return model


def window_centers(mask, np):
    """Same centres as the V14 reader and the publisher's _eval_centers (96 slices -> 94 windows)."""
    mask = np.asarray(mask)
    depth = int(mask.shape[0])
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    low, high = int(valid.min()), int(valid.max())
    candidates = [c for c in range(low + 1, high) if c - 1 >= low and c + 1 <= high]
    if not candidates:
        candidates = [max(1, min((low + high) // 2, depth - 2))]
    indices = np.linspace(0, len(candidates) - 1, WINDOW_COUNT).round().astype(int)
    return np.asarray([candidates[int(index)] for index in indices], dtype=np.int64)


def centre_crop(tensor):
    """Publisher _crop: round(H*0.8) centred, applied to the last two axes."""
    height, width = tensor.shape[-2], tensor.shape[-1]
    crop_h, crop_w = int(round(height * CROP_FRACTION)), int(round(width * CROP_FRACTION))
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return tensor[..., top:top + crop_h, left:left + crop_w]


def prepare_windows(volume, mask, device, np, torch):
    """Publisher gpu_windows values: LUT(uint8/255) -> crop 307 -> bilinear 224 -> ImageNet norm."""
    centers = window_centers(mask, np)
    triplets = torch.from_numpy(np.ascontiguousarray(
        volume[np.stack([centers - 1, centers, centers + 1], axis=1)])).to(device)
    require(tuple(triplets.shape) == (WINDOW_COUNT, 3, CORPUS_RES, CORPUS_RES), "R224 triplet shape differs")
    lut = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(device)
    values = lut[triplets.long()]
    values = centre_crop(values)
    values = torch.nn.functional.interpolate(values, size=(RESOLUTION, RESOLUTION),
                                             mode="bilinear", align_corners=False)
    means = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1).to(device)
    scales = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1).to(device)
    windows = (values - means) / scales
    require(tuple(windows.shape) == (WINDOW_COUNT, 3, RESOLUTION, RESOLUTION), "R224 input shape differs")
    return windows


def predict(model, windows, np, torch, cuda):
    """Return (float32 probabilities of shape (12,), attempt name). Raises the last error if all fail."""
    last = None
    for name, chunk, half in ATTEMPTS:
        try:
            with torch.no_grad():
                if half and cuda:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(windows.unsqueeze(0), chunk=chunk).float()
                else:
                    logits = model(windows.unsqueeze(0), chunk=chunk).float()
                require(bool(torch.isfinite(logits).all()), "Nonfinite R224 logits")
                probabilities = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
            require(probabilities.shape == (LABEL_COUNT,) and bool(np.isfinite(probabilities).all())
                    and bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                    "Invalid R224 probabilities")
            return probabilities, name
        except Exception as error:
            last = error
            if cuda:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    raise last


def ordinal_ranks(probabilities, np):
    """V14 and Codex rank: argsort.argsort / max(1, N-1), float32, non-finite -> 0.5."""
    values = np.asarray(probabilities)
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[0] > 0
            and values.shape[1] == LABEL_COUNT, "Expected float32 study-by-12 values")
    ranks = (values.argsort(0).argsort(0).astype(np.float64) / max(1, len(values) - 1)).astype(np.float32)
    ranks[~np.isfinite(ranks)] = 0.5
    return ranks


def blend_ranks(r384_ranks, r224_ranks, np):
    """Codex rank_blend semantics: (1-w) * primary + w * secondary, float32, w = 0.25 on r224."""
    left = np.asarray(r384_ranks, dtype=np.float32)
    right = np.asarray(r224_ranks, dtype=np.float32)
    require(left.shape == right.shape and left.ndim == 2 and left.shape[1] == LABEL_COUNT,
            "Aligned rank shapes differ")
    return np.float32(1.0 - R224_WEIGHT) * left + np.float32(R224_WEIGHT) * right


def prepare_arm(input_root, devices, timm, torch):
    """Verify bytes, then load one strict copy per device. Never raises: returns (models or None, status)."""
    status = dict(enabled=False, reason=None, checkpoint_name=CHECKPOINT_NAME, checkpoint_bytes_expected=CHECKPOINT_BYTES,
                  checkpoint_sha256_expected=CHECKPOINT_SHA256, checkpoint_sha256_verified=False,
                  checkpoint_sha256_computed=None, checkpoint_bytes_observed=None, strict_load_ok=False)
    models = []
    try:
        path = checkpoint_path(input_root)
        matches, digest, size = verify_checkpoint(path)
        status.update(checkpoint_sha256_verified=matches, checkpoint_sha256_computed=digest,
                      checkpoint_bytes_observed=size)
        require(matches, "R224 checkpoint bytes differ from the publisher SHA-256 or size")
        for device in devices:
            models.append(load_model(path, device, timm, torch))
        status.update(enabled=True, strict_load_ok=True)
        return dict(zip(devices, models)), status
    except Exception as error:
        for model in models:
            del model
        status.update(enabled=False, reason=f"{type(error).__name__}: {str(error)[:300]}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None, status
