import pytest

from backend.policy_servers.quantization import (
    LinearWeights,
    available_precisions,
    quantized_linear,
)


@pytest.mark.parametrize(
    ("capability", "has_fp8_dtype", "expected"),
    [
        ((8, 0), True, ("bf16", "int8")),
        ((8, 9), True, ("bf16", "fp8", "int8")),
        ((12, 0), True, ("bf16", "fp8", "int8")),
        ((12, 0), False, ("bf16", "int8")),
    ],
)
def test_precision_depends_on_cuda_architecture_and_dtype(capability, has_fp8_dtype, expected):
    assert available_precisions(capability, has_fp8_dtype=has_fp8_dtype) == expected


def test_pre_ampere_cannot_use_the_bf16_runtime():
    with pytest.raises(RuntimeError, match="Ampere"):
        available_precisions((7, 5), has_fp8_dtype=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch", reason="Install optional PyTorch for CPU tensor tests")


@pytest.mark.parametrize(
    ("precision", "dtype", "limit", "relative_error"),
    [("fp8", "float8_e4m3fn", 448, 0.07), ("int8", "int8", 127, 0.04)],
)
def test_weight_only_quantization_scales_each_output_channel(
    torch, precision, dtype, limit, relative_error
):
    linear = torch.nn.Linear(3, 3)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[12.0, -7.4, 1.1], [0.012, -0.0074, 0.0011], [0, 0, 0]]))
        linear.bias.copy_(torch.tensor([0.25, -0.125, 1.0]))
    quantized = quantized_linear(linear, torch, precision, "cpu")
    assert quantized.weight.dtype == getattr(torch, dtype)
    assert quantized.scale.shape == (3, 1)
    assert quantized.scale.dtype == torch.bfloat16
    assert quantized.scale[2, 0] > 0
    assert quantized.weight.float()[0, 0].item() == limit
    assert quantized.weight.float()[1, 0].item() == limit
    assert torch.count_nonzero(quantized.weight.float()[2]) == 0

    x = torch.eye(3)
    expected = linear(x).detach()
    result = quantized(x)
    assert result.dtype == torch.bfloat16
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), expected, rtol=relative_error, atol=0.002)


def test_select_precision_keeps_other_layers_and_restores_bf16_weights(torch):
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.mlp = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    model.attention = torch.nn.Linear(2, 2)
    original_mlp = model.block.mlp[0]
    original_attention = model.attention
    variants = LinearWeights(model, torch, "cpu", projections=())

    with torch.inference_mode():
        variants.select("fp8")
        assert model.block.mlp[0].weight.dtype == torch.float8_e4m3fn
        variants.select("int8")
        assert model.block.mlp[0].weight.dtype == torch.int8
        variants.select("bf16")
    assert model.attention is original_attention
    assert model.block.mlp[0] is original_mlp
