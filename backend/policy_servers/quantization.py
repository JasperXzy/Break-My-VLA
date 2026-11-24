"""Per-output-channel weight quantization for VLA MLPs and projections."""

from backend.shared.schemas import Precision


def available_precisions(
    capability: tuple[int, int], *, has_fp8_dtype: bool
) -> tuple[Precision, ...]:
    if capability < (8, 0):
        raise RuntimeError("BF16 requires an NVIDIA Ampere or newer GPU")
    if capability >= (8, 9) and has_fp8_dtype:
        return ("bf16", "fp8", "int8")
    return ("bf16", "int8")


def quantized_linear(linear, torch, precision: Precision, device):
    """Store scaled weights and dequantize to BF16 for the linear operation."""

    class WeightOnlyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = linear.in_features
            self.out_features = linear.out_features
            if precision == "int8":
                dtype, limit = torch.int8, 127
            elif precision == "fp8":
                dtype = torch.float8_e4m3fn
                limit = torch.finfo(dtype).max
            else:
                raise ValueError(f"Unsupported weight quantization: {precision}")
            weight = linear.weight.detach().to(device=device, dtype=torch.float32)
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / limit
            scaled = (weight / scale).clamp(-limit, limit)
            if precision == "int8":
                scaled = scaled.round()
            self.register_buffer("weight", scaled.to(dtype))
            self.register_buffer("scale", scale.to(torch.bfloat16))
            bias = linear.bias
            if bias is not None:
                bias = bias.detach().to(device=device, dtype=torch.bfloat16)
            self.register_buffer("bias", bias)

        def forward(self, activations):
            weight = self.weight.to(self.scale.dtype) * self.scale
            return torch.nn.functional.linear(activations.to(weight.dtype), weight, self.bias)

    return WeightOnlyLinear().eval()


class LinearWeights:
    """Keep one active set of target weights on device and retain the originals on CPU."""

    def __init__(
        self, model, torch, device, *, projections: tuple[str, ...], prefixes: tuple[str, ...] = ()
    ):
        self._torch = torch
        self._device = device
        self._precision: Precision = "bf16"
        self._targets = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            selected = name.startswith(prefixes) if prefixes else ".mlp." in name
            if selected or name in projections:
                parent, _, child = name.rpartition(".")
                self._targets.append((model.get_submodule(parent), child, module))
        if not self._targets:
            raise RuntimeError("The checkpoint has no matching MLP or projection Linear layers")

    def select(self, precision: Precision) -> None:
        if precision == self._precision:
            return

        originals = [linear for _, _, linear in self._targets]
        if precision == "bf16":
            try:
                replacements = [linear.to(self._device) for linear in originals]
            except RuntimeError:
                for linear in originals:
                    linear.cpu()
                raise
        else:
            replacements = [
                quantized_linear(linear, self._torch, precision, self._device)
                for linear in originals
            ]

        for (parent, name, _), replacement in zip(self._targets, replacements, strict=True):
            setattr(parent, name, replacement)
        if self._precision == "bf16":
            for linear in originals:
                linear.cpu()
        self._precision = precision
