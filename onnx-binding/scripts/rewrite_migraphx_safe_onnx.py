#!/usr/bin/env python3
"""Rewrite ONNX patterns that block MIGraphX ownership.

This is intentionally an artifact-preparation tool, not a runtime mutation.
Run it after exporting an ONNX artifact and before uploading or benchmarking the
MIGraphX candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper, shape_inference

SKIP_LAYER_NORM_MIN_INPUTS = 3
SKIP_LAYER_NORM_BETA_INDEX = 3
SKIP_LAYER_NORM_BIAS_INDEX = 4
WHERE_INPUT_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite known ONNX patterns into MIGraphX-safe equivalents."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX file.")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX file.")
    parser.add_argument(
        "--rewrite-skip-layer-normalization",
        action="store_true",
        help=(
            "Rewrite com.microsoft::SkipLayerNormalization into standard "
            "Add + LayerNormalization. This avoids MIGraphX 2.15 rejecting "
            "SkipLayerNormalization nodes with an empty beta input."
        ),
    )
    parser.add_argument(
        "--rewrite-com-ms-gelu",
        action="store_true",
        help=(
            "Rewrite com.microsoft::Gelu into primitive ONNX ops. This removes "
            "the remaining contrib-domain nodes that can prevent ORT MIGraphX "
            "EP from claiming otherwise compilable SDPA sequence artifacts."
        ),
    )
    parser.add_argument(
        "--rewrite-split-num-outputs",
        action="store_true",
        help=(
            "Rewrite Split-18 nodes that use the num_outputs attribute into "
            "the opset-17-compatible explicit split-input form."
        ),
    )
    parser.add_argument(
        "--rewrite-softmax-nan-guard",
        action="store_true",
        help=(
            "Rewrite Softmax -> IsNaN -> Where(zero, softmax) guards into a "
            "direct Softmax edge. This is valid only for classifier artifacts "
            "whose validation inputs always contain at least one unmasked token."
        ),
    )
    parser.add_argument(
        "--strip-unused-opset-imports",
        action="store_true",
        help="Remove opset imports for domains that no node uses.",
    )
    parser.add_argument(
        "--target-default-opset",
        type=int,
        help="Set the default ONNX opset version after rewrites, for example 17.",
    )
    parser.add_argument(
        "--fail-if-unchanged",
        action="store_true",
        help="Exit non-zero if no node was rewritten.",
    )
    return parser.parse_args()


def make_fp16_scalar_constant(name: str, output: str, value: float) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=helper.make_tensor(
            name=f"{name}_value",
            data_type=TensorProto.FLOAT16,
            dims=[],
            vals=[value],
        ),
    )


def make_i64_vector_constant(
    name: str, output: str, values: list[int]
) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=helper.make_tensor(
            name=f"{name}_value",
            data_type=TensorProto.INT64,
            dims=[len(values)],
            vals=values,
        ),
    )


def rewrite_skip_layer_normalization(model: onnx.ModelProto) -> int:
    rewritten = 0
    nodes = []
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "SkipLayerNormalization":
            if len(node.input) < SKIP_LAYER_NORM_MIN_INPUTS:
                raise ValueError(
                    f"{node.name or node.op_type} has {len(node.input)} inputs; "
                    "expected at least input, skip, and gamma"
                )

            hidden, skip, gamma = node.input[:SKIP_LAYER_NORM_MIN_INPUTS]
            beta = (
                node.input[SKIP_LAYER_NORM_BETA_INDEX]
                if len(node.input) > SKIP_LAYER_NORM_BETA_INDEX
                else ""
            )
            bias = (
                node.input[SKIP_LAYER_NORM_BIAS_INDEX]
                if len(node.input) > SKIP_LAYER_NORM_BIAS_INDEX
                else ""
            )
            base_name = node.name or "SkipLayerNormalization"
            residual = f"{base_name}_migraphx_residual"
            add_inputs = [hidden, skip]
            if bias:
                biased = f"{base_name}_migraphx_bias"
                nodes.append(
                    helper.make_node(
                        "Add",
                        add_inputs,
                        [biased],
                        name=f"{base_name}_MIGraphX_AddSkip",
                    )
                )
                add_inputs = [biased, bias]
            nodes.append(
                helper.make_node(
                    "Add",
                    add_inputs,
                    [residual],
                    name=f"{base_name}_MIGraphX_Add",
                )
            )
            ln_inputs = [residual, gamma]
            if beta:
                ln_inputs.append(beta)
            attrs = {
                attr.name: helper.get_attribute_value(attr) for attr in node.attribute
            }
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    ln_inputs,
                    list(node.output),
                    name=f"{base_name}_MIGraphX_LayerNormalization",
                    **attrs,
                )
            )
            rewritten += 1
        else:
            nodes.append(node)

    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return rewritten


def rewrite_com_ms_gelu(model: onnx.ModelProto) -> int:
    rewritten = 0
    nodes = []
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type == "Gelu":
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError(
                    f"{node.name or node.op_type} has {len(node.input)} inputs and "
                    f"{len(node.output)} outputs; expected one input and one output"
                )
            if node.attribute:
                attrs = ", ".join(attr.name for attr in node.attribute)
                raise ValueError(
                    f"{node.name or node.op_type} has unsupported attributes: {attrs}"
                )

            x = node.input[0]
            y = node.output[0]
            base_name = node.name or "Gelu"
            inv_sqrt2 = f"{base_name}_migraphx_inv_sqrt2"
            half = f"{base_name}_migraphx_half"
            one = f"{base_name}_migraphx_one"
            scaled = f"{base_name}_migraphx_scaled"
            erf = f"{base_name}_migraphx_erf"
            erf_plus_one = f"{base_name}_migraphx_erf_plus_one"
            weighted = f"{base_name}_migraphx_weighted"

            nodes.extend(
                [
                    make_fp16_scalar_constant(
                        f"{base_name}_MIGraphX_InvSqrt2",
                        inv_sqrt2,
                        0.7071067811865476,
                    ),
                    make_fp16_scalar_constant(
                        f"{base_name}_MIGraphX_Half",
                        half,
                        0.5,
                    ),
                    make_fp16_scalar_constant(
                        f"{base_name}_MIGraphX_One",
                        one,
                        1.0,
                    ),
                    helper.make_node(
                        "Mul",
                        [x, inv_sqrt2],
                        [scaled],
                        name=f"{base_name}_MIGraphX_Scale",
                    ),
                    helper.make_node(
                        "Erf",
                        [scaled],
                        [erf],
                        name=f"{base_name}_MIGraphX_Erf",
                    ),
                    helper.make_node(
                        "Add",
                        [erf, one],
                        [erf_plus_one],
                        name=f"{base_name}_MIGraphX_AddOne",
                    ),
                    helper.make_node(
                        "Mul",
                        [x, erf_plus_one],
                        [weighted],
                        name=f"{base_name}_MIGraphX_Weight",
                    ),
                    helper.make_node(
                        "Mul",
                        [weighted, half],
                        [y],
                        name=f"{base_name}_MIGraphX_Gelu",
                    ),
                ]
            )
            rewritten += 1
        else:
            nodes.append(node)

    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return rewritten


def tensor_shape_by_name(model: onnx.ModelProto) -> dict[str, list[Any]]:
    inferred = shape_inference.infer_shapes(model)
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    )
    shapes: dict[str, list[Any]] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[Any] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("")
        shapes[value.name] = dims
    return shapes


def rewrite_split_num_outputs(model: onnx.ModelProto) -> int:
    shapes = tensor_shape_by_name(model)
    rewritten = 0
    nodes = []
    for node in model.graph.node:
        if node.op_type != "Split":
            nodes.append(node)
            continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if "num_outputs" not in attrs:
            nodes.append(node)
            continue
        if len(node.input) != 1:
            raise ValueError(
                f"{node.name or node.op_type} uses num_outputs but has "
                f"{len(node.input)} inputs; expected one input"
            )

        axis = int(attrs.get("axis", 0))
        split_sizes: list[int] = []
        for output in node.output:
            shape = shapes.get(output)
            if shape is None:
                raise ValueError(
                    f"{node.name or node.op_type} output {output} has no inferred shape"
                )
            normalized_axis = axis if axis >= 0 else len(shape) + axis
            if normalized_axis < 0 or normalized_axis >= len(shape):
                raise ValueError(
                    f"{node.name or node.op_type} axis {axis} is invalid for {output} shape {shape}"
                )
            size = shape[normalized_axis]
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"{node.name or node.op_type} output {output} has non-static "
                    f"split size {size!r} on axis {axis}"
                )
            split_sizes.append(size)

        base_name = node.name or "Split"
        split_input = f"{base_name}_migraphx_split_sizes"
        nodes.append(
            make_i64_vector_constant(
                f"{base_name}_MIGraphX_SplitSizes",
                split_input,
                split_sizes,
            )
        )
        kept_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
            if attr.name != "num_outputs"
        }
        nodes.append(
            helper.make_node(
                "Split",
                [node.input[0], split_input],
                list(node.output),
                name=f"{base_name}_MIGraphX_Split",
                **kept_attrs,
            )
        )
        rewritten += 1

    if rewritten:
        del model.graph.node[:]
        model.graph.node.extend(nodes)
    return rewritten


def rewrite_softmax_nan_guard(model: onnx.ModelProto) -> int:
    producer_by_output = {
        output: node for node in model.graph.node for output in node.output
    }
    replacement_by_output: dict[str, str] = {}
    remove_node_ids: set[int] = set()
    rewritten = 0

    for node in model.graph.node:
        if (
            node.op_type != "Where"
            or len(node.input) != WHERE_INPUT_COUNT
            or len(node.output) != 1
        ):
            continue

        condition, _, fallback = node.input
        condition_producer = producer_by_output.get(condition)
        fallback_producer = producer_by_output.get(fallback)
        if (
            condition_producer is None
            or condition_producer.op_type != "IsNaN"
            or len(condition_producer.input) != 1
            or condition_producer.input[0] != fallback
            or fallback_producer is None
            or fallback_producer.op_type != "Softmax"
        ):
            continue

        replacement_by_output[node.output[0]] = fallback
        remove_node_ids.add(id(node))
        remove_node_ids.add(id(condition_producer))
        rewritten += 1

    if not rewritten:
        return 0

    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            replacement = input_name
            while replacement in replacement_by_output:
                replacement = replacement_by_output[replacement]
            node.input[index] = replacement

    nodes = [node for node in model.graph.node if id(node) not in remove_node_ids]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return rewritten


def strip_unused_opset_imports(model: onnx.ModelProto) -> int:
    used_domains = {node.domain for node in model.graph.node}
    previous = list(model.opset_import)
    kept = [opset for opset in previous if opset.domain in used_domains]
    if len(kept) == len(previous):
        return 0
    del model.opset_import[:]
    model.opset_import.extend(kept)
    return len(previous) - len(kept)


def set_default_opset(model: onnx.ModelProto, version: int) -> None:
    found = False
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = version
            found = True
            break
    if not found:
        model.opset_import.append(helper.make_opsetid("", version))


def main() -> None:
    args = parse_args()
    if not any(
        [
            args.rewrite_skip_layer_normalization,
            args.rewrite_com_ms_gelu,
            args.rewrite_split_num_outputs,
            args.rewrite_softmax_nan_guard,
            args.strip_unused_opset_imports,
            args.target_default_opset,
        ]
    ):
        raise SystemExit("No rewrite selected.")

    model = onnx.load(args.input, load_external_data=True)
    rewrites = 0
    if args.rewrite_skip_layer_normalization:
        rewrites += rewrite_skip_layer_normalization(model)
    if args.rewrite_com_ms_gelu:
        rewrites += rewrite_com_ms_gelu(model)
    if args.rewrite_split_num_outputs:
        rewrites += rewrite_split_num_outputs(model)
    if args.rewrite_softmax_nan_guard:
        rewrites += rewrite_softmax_nan_guard(model)
    if args.strip_unused_opset_imports:
        rewrites += strip_unused_opset_imports(model)
    if args.target_default_opset:
        set_default_opset(model, args.target_default_opset)

    if args.fail_if_unchanged and rewrites == 0:
        raise SystemExit("No nodes were rewritten.")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(
        f"Wrote {args.output} with {rewrites} MIGraphX-safe rewrite"
        f"{'' if rewrites == 1 else 's'}."
    )


if __name__ == "__main__":
    main()
