"""
test_vmramar.py — local shape/forward test for VMRAMaR.

Run from the directory containing your vmramar/ package:
    python test_vmramar.py

Stubs out the Mirai backbone and common_parts so you don't need
the full Mirai codebase to run the test.
"""
import sys
import types
import torch
import torch.nn as nn

# ── 1. Stub out Mirai-specific imports ────────────────────────────────────
# This lets us test the VMRAMaR code without the onconet codebase.

def make_stub_encoder(out_channels=512, out_h=64, out_w=52):
    """Fake image encoder that returns fixed-size feature maps."""
    class StubEncoder(nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, out_channels, out_h, out_w)
    return StubEncoder()


class StubCumulativeProbabilityLayer(nn.Module):
    """Fake AHL that outputs 5-year risk logits."""
    def __init__(self, input_dim, max_followup=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, max_followup)

    def forward(self, x):
        return self.fc(x)


# Inject stub common_parts module so vmramar.model can import from it
stub_common = types.ModuleType("vmramar.common_parts")
stub_common.CumulativeProbabilityLayer = StubCumulativeProbabilityLayer
stub_common.extract_mirai_backbone     = lambda path: make_stub_encoder()
sys.modules["common_parts"] = stub_common

# Stub config
stub_cfg_mod = types.ModuleType("vmramar.config")
stub_cfg_mod.cfg = {
    "paths": {
        "asymMirai_master_onconet": "/tmp",
        "mirai_path": "/tmp/fake.p",
    }
}
sys.modules["config"]       = stub_cfg_mod

# ── 2. Now import the real VMRAMaR code ───────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vmrnn            import VMRNN
from image_aggregator import ImageAggregator
from sad              import SpatialAsymmetryDetector
from lat              import LongitudinalAsymmetryTracker
from model            import VMRAMaR

# ── 3. Minimal args ───────────────────────────────────────────────────────

class Args:
    embed_dim            = 512
    num_images           = 4           # CC_L, CC_R, MLO_L, MLO_R
    depths_downsample    = [2, 2]
    depths_upsample      = [2, 2]
    freeze_image_encoder = False
    use_asymmetry        = True
    latent_h             = 5
    latent_w             = 5
    use_sad_bias         = True
    use_sad_bn           = False       # off so we don't need large batch for BN
    use_lat_bn           = False
    lat_dropout          = 0.0


args = Args()

# ── 4. Helper ─────────────────────────────────────────────────────────────

def check(name, tensor, expected_shape):
    ok = tuple(tensor.shape) == tuple(expected_shape)
    status = "✅" if ok else "❌"
    print(f"  {status}  {name}: {tuple(tensor.shape)}  (expected {tuple(expected_shape)})")
    if not ok:
        raise AssertionError(f"{name} shape mismatch")


# ── 5. Unit tests ─────────────────────────────────────────────────────────

def test_vmrnn():
    print("\n── VMRNN (temporal mode) ──────────────────────────────────────")
    B, T, C = 2, 5, 512
    vmrnn = VMRNN(
        embed_dim=C,
        depths_downsample=args.depths_downsample,
        depths_upsample=args.depths_upsample,
        feature_resolution=(1, 1),
    )
    x = torch.randn(B, T, C)
    out, sd, su = vmrnn(x)
    check("output", out, (B, T, C))
    print(f"  ✅  states_down: {len(sd)} levels")
    print(f"  ✅  states_up:   {len(su)} levels")


def test_image_aggregator():
    print("\n── ImageAggregator ────────────────────────────────────────────")
    B, T, V, C = 2, 5, 4, 512
    agg = ImageAggregator(C, num_views=V)
    x   = torch.randn(B, T, V, C)
    out = agg(x)
    check("output", out, (B, T, C))


def test_sad():
    print("\n── SpatialAsymmetryDetector ───────────────────────────────────")
    B, T, C, H, W = 2, 5, 512, 64, 52
    sad  = SpatialAsymmetryDetector(args)
    left  = torch.randn(B, T, C, H, W)
    right = torch.randn(B, T, C, H, W)
    out  = sad(left, right)
    check("asymmetry_values", out["asymmetry_values"], (B, T))
    check("asymmetry_coords", out["asymmetry_coords"], (B, T, 2))
    lh, lw = args.latent_h, args.latent_w
    check("heatmap",          out["heatmap"],          (B, T, lh, lw))


def test_lat():
    print("\n── LongitudinalAsymmetryTracker ───────────────────────────────")
    B, T, D = 2, 5, 512
    lh, lw  = args.latent_h, args.latent_w
    lat      = LongitudinalAsymmetryTracker(args)
    feats    = torch.randn(B, T, D)
    coords   = torch.randint(0, 5, (B, T, 2)).float()
    heatmaps = torch.randn(B, T, lh, lw)
    out      = lat(feats, coords, heatmaps)
    check("output", out, (B, D))


def test_full_model():
    print("\n── Full VMRAMaR forward ───────────────────────────────────────")
    B, T, C, V, H, W = 2, 5, 1, 4, 256, 256
    model = VMRAMaR(args)
    model.eval()

    # Fake batch — images (B, T, C, V, H, W)
    images = torch.randn(B, T, C, V, H, W)
    target = torch.zeros(B, 5)
    y_mask = torch.ones(B, 5)

    batch = {"images": images, "target": target, "y_mask": y_mask}

    with torch.no_grad():
        outputs = model(batch)

    logit = outputs["logit"]
    check("logit", logit, (B, 5))

    # Check risk heads
    heads = model.get_risk_heads(outputs, batch)
    assert "logit_output" in heads
    print("  ✅  get_risk_heads OK")

    primary = model.get_primary_risk_head(outputs)
    check("primary_risk_head", primary, (B, 5))


def test_no_asymmetry():
    print("\n── VMRAMaR without asymmetry ──────────────────────────────────")
    a = Args()
    a.use_asymmetry = False
    a.use_sad_bn    = False
    a.use_lat_bn    = False
    B, T, C, V, H, W = 2, 3, 1, 4, 128, 128
    model  = VMRAMaR(a)
    images = torch.randn(B, T, C, V, H, W)
    batch  = {"images": images, "target": torch.zeros(B, 5), "y_mask": torch.ones(B, 5)}
    with torch.no_grad():
        out = model(batch)
    check("logit (no asym)", out["logit"], (B, 5))


def test_variable_timesteps():
    print("\n── Variable number of timesteps ───────────────────────────────")
    model = VMRAMaR(args)
    model.eval()
    for T in [1, 3, 5]:
        images = torch.randn(2, T, 1, 4, 128, 128)
        batch  = {"images": images, "target": torch.zeros(2, 5), "y_mask": torch.ones(2, 5)}
        with torch.no_grad():
            out = model(batch)
        check(f"logit T={T}", out["logit"], (2, 5))


# ── 6. Run all tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("VMRAMaR local shape tests")
    print("=" * 60)

    try:
        test_vmrnn()
        test_image_aggregator()
        test_sad()
        test_lat()
        test_full_model()
        test_no_asymmetry()
        test_variable_timesteps()
        print("\n" + "=" * 60)
        print("✅  All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌  FAILED: {e}")
        sys.exit(1)