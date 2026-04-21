"""Helpers for ONNX node counts and a few simple graph metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import prod

import onnx
from onnx import AttributeProto, shape_inference

INPUT_DIMENSIONS_COLUMN = "input_dimensions"
INPUT_DTYPES_COLUMN = "input_dtypes"
OUTPUT_DIMENSIONS_COLUMN = "output_dimensions"
OUTPUT_DTYPES_COLUMN = "output_dtypes"
CONV_FLOPS_COLUMN = "conv_flops"
MATMUL_FLOPS_COLUMN = "matmul_flops"
ELEMENTWISE_BYTES_COLUMN = "elementwise_bytes"
REDUCTION_BYTES_COLUMN = "reduction_bytes"
NORMALIZATION_BYTES_COLUMN = "normalization_bytes"
MOVEMENT_BYTES_COLUMN = "movement_bytes"

TEXT_COLUMNS = [
    INPUT_DIMENSIONS_COLUMN,
    INPUT_DTYPES_COLUMN,
    OUTPUT_DIMENSIONS_COLUMN,
    OUTPUT_DTYPES_COLUMN,
]

FIXED_COLUMNS = [
    *TEXT_COLUMNS,
    CONV_FLOPS_COLUMN,
    MATMUL_FLOPS_COLUMN,
    ELEMENTWISE_BYTES_COLUMN,
    REDUCTION_BYTES_COLUMN,
    NORMALIZATION_BYTES_COLUMN,
    MOVEMENT_BYTES_COLUMN,
]

CONV_OPS = {"Conv", "QLinearConv", "ConvTranspose"}
MATMUL_OPS = {"MatMul", "Gemm", "QLinearMatMul", "Attention", "Einsum", "GRU", "LSTM", "RNN"}
ELEMENTWISE_OPS = {
    "Relu",
    "Sigmoid",
    "Tanh",
    "LeakyRelu",
    "Gelu",
    "Swish",
    "HardSwish",
    "Add",
    "Mul",
    "Div",
    "Sub",
    "Clip",
    "Abs",
    "Neg",
    "Pow",
    "Where",
    "Sqrt",
    "HardSigmoid",
    "Selu",
    "PRelu",
    "Mish",
    "Reciprocal",
    "Ceil",
    "Floor",
    "Mod",
    "Max",
    "Greater",
    "Exp",
    "Log",
}
REDUCTION_OPS = {
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "TopK",
    "ReduceLogSumExp",
    "ReduceL1",
    "ReduceL2",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
    "NonMaxSuppression",
    "ArgMax",
    "ArgMin",
}
NORMALIZATION_OPS = {
    "Softmax",
    "LogSoftmax",
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "RMSNormalization",
}
MOVEMENT_OPS = {
    "Transpose",
    "Reshape",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Split",
    "Slice",
    "Gather",
    "GatherND",
    "Expand",
    "Tile",
    "Resize",
    "Upsample",
    "Pad",
    "Scatter",
    "ScatterND",
    "GatherElements",
    "ScatterElements",
    "TensorScatter",
}


def build_dtype_maps() -> tuple[dict[int, str], dict[int, float]]:
    specs = {
        "BOOL": ("bool", 1.0),
        "UINT4": ("uint4", 0.5),
        "INT4": ("int4", 0.5),
        "FLOAT4E2M1": ("float4e2m1", 0.5),
        "UINT8": ("uint8", 1.0),
        "INT8": ("int8", 1.0),
        "FLOAT8E4M3FN": ("float8e4m3fn", 1.0),
        "FLOAT8E4M3FNUZ": ("float8e4m3fnuz", 1.0),
        "FLOAT8E5M2": ("float8e5m2", 1.0),
        "FLOAT8E5M2FNUZ": ("float8e5m2fnuz", 1.0),
        "UINT16": ("uint16", 2.0),
        "INT16": ("int16", 2.0),
        "FLOAT16": ("float16", 2.0),
        "BFLOAT16": ("bfloat16", 2.0),
        "UINT32": ("uint32", 4.0),
        "INT32": ("int32", 4.0),
        "FLOAT": ("float32", 4.0),
        "UINT64": ("uint64", 8.0),
        "INT64": ("int64", 8.0),
        "DOUBLE": ("float64", 8.0),
        "COMPLEX64": ("complex64", 8.0),
        "COMPLEX128": ("complex128", 16.0),
        "STRING": ("string", 0.0),
    }
    names: dict[int, str] = {}
    sizes: dict[int, float] = {}
    for proto_name, (display_name, size) in specs.items():
        enum_value = getattr(onnx.TensorProto, proto_name, None)
        if enum_value is None:
            continue
        names[enum_value] = display_name
        sizes[enum_value] = size
    return names, sizes


DTYPE_NAMES, DTYPE_SIZES = build_dtype_maps()


@dataclass
class TensorInfo:
    shape: tuple[int, ...] | None
    elem_type: int | None = None
    int_values: tuple[int, ...] | None = None


def collect_model_row(model: onnx.ModelProto, model_name: str) -> dict[str, str]:
    model = infer_shapes(model)
    graph = model.graph
    tensor_map = build_tensor_map(graph)
    counts: Counter[str] = Counter()
    totals = {
        CONV_FLOPS_COLUMN: 0,
        MATMUL_FLOPS_COLUMN: 0,
        ELEMENTWISE_BYTES_COLUMN: 0.0,
        REDUCTION_BYTES_COLUMN: 0.0,
        NORMALIZATION_BYTES_COLUMN: 0.0,
        MOVEMENT_BYTES_COLUMN: 0.0,
    }

    collect_graph_metrics(graph, tensor_map, counts, totals)

    row = {
        "model": model_name,
        INPUT_DIMENSIONS_COLUMN: format_graph_shapes(graph.input, graph.initializer, tensor_map),
        INPUT_DTYPES_COLUMN: format_graph_dtypes(graph.input, graph.initializer, tensor_map),
        OUTPUT_DIMENSIONS_COLUMN: format_graph_shapes(graph.output, (), tensor_map),
        OUTPUT_DTYPES_COLUMN: format_graph_dtypes(graph.output, (), tensor_map),
        CONV_FLOPS_COLUMN: format_number(totals[CONV_FLOPS_COLUMN]),
        MATMUL_FLOPS_COLUMN: format_number(totals[MATMUL_FLOPS_COLUMN]),
        ELEMENTWISE_BYTES_COLUMN: format_number(totals[ELEMENTWISE_BYTES_COLUMN]),
        REDUCTION_BYTES_COLUMN: format_number(totals[REDUCTION_BYTES_COLUMN]),
        NORMALIZATION_BYTES_COLUMN: format_number(totals[NORMALIZATION_BYTES_COLUMN]),
        MOVEMENT_BYTES_COLUMN: format_number(totals[MOVEMENT_BYTES_COLUMN]),
    }

    for node_type in sorted(counts, key=str.casefold):
        row[node_type] = str(counts[node_type])

    return row


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        return shape_inference.infer_shapes(model)
    except Exception:
        return model


def collect_graph_metrics(
    graph: onnx.GraphProto,
    tensor_map: dict[str, TensorInfo],
    counts: Counter[str],
    totals: dict[str, int | float],
) -> None:
    for node in graph.node:
        update_output_tensor_map(node, tensor_map)

        counts[format_node_type(node)] += 1
        total_bytes = sum_known_bytes(tensor_map, node.input) + sum_known_bytes(tensor_map, node.output)

        if node.op_type in CONV_OPS:
            totals[CONV_FLOPS_COLUMN] += estimate_conv_flops(node, tensor_map)
        elif node.op_type in MATMUL_OPS:
            totals[MATMUL_FLOPS_COLUMN] += estimate_matmul_flops(node, tensor_map)
        elif node.op_type in ELEMENTWISE_OPS:
            totals[ELEMENTWISE_BYTES_COLUMN] += total_bytes
        elif node.op_type in REDUCTION_OPS:
            totals[REDUCTION_BYTES_COLUMN] += total_bytes
        elif node.op_type in NORMALIZATION_OPS:
            totals[NORMALIZATION_BYTES_COLUMN] += total_bytes
        elif node.op_type in MOVEMENT_OPS:
            totals[MOVEMENT_BYTES_COLUMN] += total_bytes

        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                child_tensor_map = dict(tensor_map)
                child_tensor_map.update(build_tensor_map(attribute.g))
                collect_graph_metrics(attribute.g, child_tensor_map, counts, totals)
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    child_tensor_map = dict(tensor_map)
                    child_tensor_map.update(build_tensor_map(subgraph))
                    collect_graph_metrics(subgraph, child_tensor_map, counts, totals)


def build_tensor_map(graph: onnx.GraphProto) -> dict[str, TensorInfo]:
    tensor_map: dict[str, TensorInfo] = {}

    for initializer in graph.initializer:
        tensor_map[initializer.name] = TensorInfo(
            shape=tuple(int(dim) for dim in initializer.dims),
            elem_type=initializer.data_type,
            int_values=initializer_int_values(initializer),
        )

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_info = tensor_info_from_value_info(value_info)
        if tensor_info is not None:
            tensor_map[value_info.name] = tensor_info

    return tensor_map


def tensor_info_from_value_info(value_info: onnx.ValueInfoProto) -> TensorInfo | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    shape = tuple(shape_dim_to_value(dim) for dim in tensor_type.shape.dim)
    elem_type = tensor_type.elem_type or None
    return TensorInfo(shape=shape, elem_type=elem_type)


def shape_dim_to_value(dim: onnx.TensorShapeProto.Dimension) -> int:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return 1


def count_elements(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    return prod(shape) if shape else 1


def initializer_int_values(initializer: onnx.TensorProto) -> tuple[int, ...] | None:
    try:
        values = onnx.numpy_helper.to_array(initializer).reshape(-1)
    except Exception:
        return None
    if values.dtype.kind not in {"i", "u"}:
        return None
    return tuple(int(value) for value in values.tolist())


def update_output_tensor_map(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> None:
    output_shapes = infer_output_shapes(node, tensor_map)
    output_types = infer_output_elem_types(node, tensor_map)

    for index, output_name in enumerate(node.output):
        if not output_name:
            continue
        existing = tensor_map.get(output_name)
        # Keep ONNX shape inference results when they exist, and only fill in missing data.
        shape = existing.shape if existing and existing.shape is not None else output_shapes[index]
        elem_type = (
            existing.elem_type
            if existing and existing.elem_type is not None
            else output_types[index]
        )
        int_values = existing.int_values if existing else None
        if shape is None and elem_type is None and int_values is None and existing is None:
            continue
        tensor_map[output_name] = TensorInfo(shape=shape, elem_type=elem_type, int_values=int_values)


def infer_output_shapes(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> list[tuple[int, ...] | None]:
    default = [None] * len(node.output)

    if node.op_type in {"Conv", "QLinearConv"}:
        return repeat_shape(infer_conv_output_shape(node, tensor_map), len(node.output))
    if node.op_type == "ConvTranspose":
        return repeat_shape(infer_conv_transpose_output_shape(node, tensor_map), len(node.output))
    if node.op_type in {"MaxPool", "AveragePool"}:
        return repeat_shape(infer_pool_output_shape(node, tensor_map), len(node.output))
    if node.op_type == "GlobalAveragePool":
        input_shape = first_known_shape(tensor_map, node.input)
        if input_shape is None or len(input_shape) < 2:
            return default
        output_shape = input_shape[:2] + (1,) * max(len(input_shape) - 2, 0)
        return repeat_shape(output_shape, len(node.output))
    if node.op_type == "Transpose":
        input_shape = first_known_shape(tensor_map, node.input)
        if input_shape is None:
            return default
        perm = get_attribute_ints(node, "perm")
        if not perm:
            perm = list(range(len(input_shape) - 1, -1, -1))
        return [tuple(input_shape[index] for index in perm)]
    if node.op_type == "Flatten":
        input_shape = first_known_shape(tensor_map, node.input)
        if input_shape is None:
            return default
        axis = normalize_axis(get_attribute(node, "axis", 1), len(input_shape))
        left = prod(input_shape[:axis]) if axis > 0 else 1
        right = prod(input_shape[axis:]) if axis < len(input_shape) else 1
        return [(left, right)]
    if node.op_type == "Split":
        return infer_split_output_shapes(node, tensor_map)

    return default


def infer_output_elem_types(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> list[int | None]:
    default_type = first_known_elem_type(tensor_map, node.input)
    output_types = [default_type] * len(node.output)

    if node.op_type == "QLinearConv" and len(node.input) > 7:
        quantized_type = first_known_elem_type(tensor_map, [node.input[7]])
        if quantized_type is not None:
            output_types = [quantized_type] * len(node.output)
    elif node.op_type == "QLinearMatMul" and len(node.input) > 7:
        quantized_type = first_known_elem_type(tensor_map, [node.input[7]])
        if quantized_type is not None:
            output_types = [quantized_type] * len(node.output)

    if node.op_type == "MaxPool" and len(output_types) > 1:
        output_types[1] = onnx.TensorProto.INT64

    return output_types


def infer_conv_output_shape(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> tuple[int, ...] | None:
    input_shape, weight_shape = get_conv_input_and_weight_shapes(node, tensor_map)
    if input_shape is None or weight_shape is None or len(input_shape) < 3 or len(weight_shape) < 3:
        return None

    strides = get_attribute_ints(node, "strides") or [1] * (len(input_shape) - 2)
    dilations = get_attribute_ints(node, "dilations") or [1] * (len(input_shape) - 2)
    pads = get_attribute_ints(node, "pads") or [0] * (2 * (len(input_shape) - 2))

    spatial: list[int] = []
    for index, input_dim in enumerate(input_shape[2:]):
        kernel = weight_shape[2 + index]
        dilation = dilations[index]
        stride = strides[index]
        pad_begin = pads[index]
        pad_end = pads[index + len(input_shape) - 2]
        output_dim = ((input_dim + pad_begin + pad_end - dilation * (kernel - 1) - 1) // stride) + 1
        spatial.append(output_dim)

    return (input_shape[0], weight_shape[0], *spatial)


def infer_conv_transpose_output_shape(
    node: onnx.NodeProto,
    tensor_map: dict[str, TensorInfo],
) -> tuple[int, ...] | None:
    input_shape = first_known_shape(tensor_map, [node.input[0]])
    weight_shape = first_known_shape(tensor_map, [node.input[1]])
    if input_shape is None or weight_shape is None or len(input_shape) < 3 or len(weight_shape) < 3:
        return None

    group = max(get_attribute(node, "group", 1), 1)
    strides = get_attribute_ints(node, "strides") or [1] * (len(input_shape) - 2)
    dilations = get_attribute_ints(node, "dilations") or [1] * (len(input_shape) - 2)
    pads = get_attribute_ints(node, "pads") or [0] * (2 * (len(input_shape) - 2))
    output_padding = get_attribute_ints(node, "output_padding") or [0] * (len(input_shape) - 2)

    spatial: list[int] = []
    for index, input_dim in enumerate(input_shape[2:]):
        kernel = weight_shape[2 + index]
        dilation = dilations[index]
        stride = strides[index]
        pad_begin = pads[index]
        pad_end = pads[index + len(input_shape) - 2]
        extra = output_padding[index]
        output_dim = stride * (input_dim - 1) + extra + dilation * (kernel - 1) + 1 - pad_begin - pad_end
        spatial.append(output_dim)

    return (input_shape[0], weight_shape[1] * group, *spatial)


def infer_pool_output_shape(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> tuple[int, ...] | None:
    input_shape = first_known_shape(tensor_map, node.input)
    if input_shape is None or len(input_shape) < 3:
        return None

    strides = get_attribute_ints(node, "strides") or [1] * (len(input_shape) - 2)
    kernel_shape = get_attribute_ints(node, "kernel_shape") or [1] * (len(input_shape) - 2)
    pads = get_attribute_ints(node, "pads") or [0] * (2 * (len(input_shape) - 2))

    spatial: list[int] = []
    for index, input_dim in enumerate(input_shape[2:]):
        kernel = kernel_shape[index]
        stride = strides[index]
        pad_begin = pads[index]
        pad_end = pads[index + len(input_shape) - 2]
        output_dim = ((input_dim + pad_begin + pad_end - kernel) // stride) + 1
        spatial.append(output_dim)

    return (*input_shape[:2], *spatial)


def infer_split_output_shapes(
    node: onnx.NodeProto,
    tensor_map: dict[str, TensorInfo],
) -> list[tuple[int, ...] | None]:
    input_shape = first_known_shape(tensor_map, node.input[:1])
    if input_shape is None or not node.output:
        return [None] * len(node.output)

    axis = normalize_axis(get_attribute(node, "axis", 0), len(input_shape))
    split_sizes = get_attribute_ints(node, "split")
    if not split_sizes and len(node.input) > 1:
        split_sizes = tensor_values_as_ints(tensor_map, node.input[1])

    if not split_sizes:
        total = input_shape[axis]
        pieces = len(node.output)
        base = total // pieces if pieces else total
        split_sizes = [base] * pieces
        remainder = total - base * pieces
        for index in range(remainder):
            split_sizes[index] += 1

    output_shapes: list[tuple[int, ...] | None] = []
    for size in split_sizes[: len(node.output)]:
        output_shape = list(input_shape)
        output_shape[axis] = size
        output_shapes.append(tuple(output_shape))

    while len(output_shapes) < len(node.output):
        output_shapes.append(None)

    return output_shapes


def repeat_shape(shape: tuple[int, ...] | None, count: int) -> list[tuple[int, ...] | None]:
    if shape is None:
        return [None] * count
    return [shape] * count


def get_conv_input_and_weight_shapes(
    node: onnx.NodeProto,
    tensor_map: dict[str, TensorInfo],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    if node.op_type == "QLinearConv":
        input_name = node.input[0]
        weight_name = node.input[3]
    else:
        input_name = node.input[0]
        weight_name = node.input[1]

    return (
        first_known_shape(tensor_map, [input_name]),
        first_known_shape(tensor_map, [weight_name]),
    )


def estimate_conv_flops(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> int:
    input_shape, weight_shape = get_conv_input_and_weight_shapes(node, tensor_map)
    output_shape = first_known_shape(tensor_map, node.output)
    if input_shape is None or weight_shape is None or output_shape is None:
        return 0

    group = max(get_attribute(node, "group", 1), 1)
    batch = output_shape[0] if len(output_shape) >= 1 else 1
    input_channels = input_shape[1] if len(input_shape) >= 2 else 1
    output_channels = output_shape[1] if len(output_shape) >= 2 else 1
    output_spatial = prod(output_shape[2:]) if len(output_shape) > 2 else 1
    kernel_elements = prod(weight_shape[2:]) if len(weight_shape) > 2 else 1
    input_channels_per_group = max(input_channels // group, 1)

    return int(2 * batch * output_spatial * input_channels_per_group * output_channels * kernel_elements)


def estimate_matmul_flops(node: onnx.NodeProto, tensor_map: dict[str, TensorInfo]) -> int:
    output_shape = first_known_shape(tensor_map, node.output)

    if node.op_type == "Attention":
        if output_shape is None:
            return 0
        sequence_length = output_shape[-2] if len(output_shape) >= 2 else 1
        hidden_size = output_shape[-1] if output_shape else 1
        return int(
            4 * sequence_length * sequence_length * hidden_size
            + 8 * sequence_length * hidden_size * hidden_size
        )

    if node.op_type == "QLinearMatMul":
        a_shape = first_known_shape(tensor_map, [node.input[0]])
        b_shape = first_known_shape(tensor_map, [node.input[3]])
    else:
        a_shape = first_known_shape(tensor_map, [node.input[0]])
        b_shape = first_known_shape(tensor_map, [node.input[1]])

    if a_shape is None or b_shape is None or output_shape is None:
        return 0

    batch_for_vector_output = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    batch_for_matrix_output = prod(output_shape[:-2]) if len(output_shape) > 2 else 1
    m = output_shape[-2] if len(output_shape) >= 2 else 1
    n = output_shape[-1] if output_shape else 1

    if node.op_type == "Gemm":
        trans_a = get_attribute(node, "transA", 0)
        k = a_shape[-2] if trans_a and len(a_shape) >= 2 else a_shape[-1]
        return int(2 * batch_for_matrix_output * m * n * k)

    if len(a_shape) == 1 and len(b_shape) == 1:
        return int(2 * a_shape[0])
    if len(a_shape) == 1:
        k = a_shape[0]
        return int(2 * batch_for_vector_output * n * k)
    if len(b_shape) == 1:
        vector_output = output_shape[-1] if output_shape else (a_shape[-2] if len(a_shape) >= 2 else 1)
        k = a_shape[-1]
        return int(2 * batch_for_vector_output * vector_output * k)

    k = a_shape[-1]
    return int(2 * batch_for_matrix_output * m * n * k)


def format_graph_shapes(value_infos, initializers, tensor_map: dict[str, TensorInfo]) -> str:
    initializer_names = {initializer.name for initializer in initializers}
    parts: list[str] = []
    for value_info in value_infos:
        if value_info.name in initializer_names:
            continue
        tensor_info = tensor_map.get(value_info.name)
        parts.append(f"{value_info.name}:{format_shape(tensor_info.shape if tensor_info else None)}")
    return "; ".join(parts)


def format_graph_dtypes(value_infos, initializers, tensor_map: dict[str, TensorInfo]) -> str:
    initializer_names = {initializer.name for initializer in initializers}
    parts: list[str] = []
    for value_info in value_infos:
        if value_info.name in initializer_names:
            continue
        tensor_info = tensor_map.get(value_info.name)
        parts.append(f"{value_info.name}:{format_dtype(tensor_info.elem_type if tensor_info else None)}")
    return "; ".join(parts)


def format_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "?"
    if not shape:
        return "scalar"
    return "x".join(str(dim) for dim in shape)


def format_dtype(elem_type: int | None) -> str:
    if elem_type is None:
        return "?"
    return DTYPE_NAMES.get(elem_type, f"type_{elem_type}")


def format_node_type(node: onnx.NodeProto) -> str:
    if node.domain not in ("", "ai.onnx"):
        return f"{node.domain}::{node.op_type}"
    return node.op_type


def format_number(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def bytes_for_tensor_info(tensor_info: TensorInfo | None) -> float | None:
    if tensor_info is None or tensor_info.elem_type is None:
        return None
    element_count = count_elements(tensor_info.shape)
    dtype_size = DTYPE_SIZES.get(tensor_info.elem_type)
    if element_count is None or dtype_size is None:
        return None
    return element_count * dtype_size


def sum_known_bytes(tensor_map: dict[str, TensorInfo], names) -> float:
    total = 0.0
    for name in names:
        value = bytes_for_tensor_info(tensor_map.get(name))
        if value is not None:
            total += value
    return total


def first_known_shape(tensor_map: dict[str, TensorInfo], names) -> tuple[int, ...] | None:
    for name in names:
        tensor_info = tensor_map.get(name)
        if tensor_info is None or tensor_info.shape is None:
            continue
        return tensor_info.shape
    return None


def first_known_elem_type(tensor_map: dict[str, TensorInfo], names) -> int | None:
    for name in names:
        tensor_info = tensor_map.get(name)
        if tensor_info is None or tensor_info.elem_type is None:
            continue
        return tensor_info.elem_type
    return None


def normalize_axis(axis: int, rank: int) -> int:
    return axis if axis >= 0 else axis + rank


def get_attribute(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    return default


def get_attribute_ints(node: onnx.NodeProto, name: str) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            value = onnx.helper.get_attribute_value(attribute)
            return [int(item) for item in value]
    return []


def tensor_values_as_ints(tensor_map: dict[str, TensorInfo], name: str) -> list[int]:
    tensor_info = tensor_map.get(name)
    if tensor_info is None or tensor_info.int_values is None:
        return []
    return list(tensor_info.int_values)
