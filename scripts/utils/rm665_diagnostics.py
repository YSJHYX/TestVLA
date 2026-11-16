"""Degree-valued logging for RM65 + SG100; model/data tensors stay in radians."""

import math
import re

import torch
import torch.distributed as dist

from utils.metric import _denormalize_tokenizer_action_batch


GROUP_NAMES = {
    "arm": "rm65_arm_6d",
    "hand_thumb_index": "sg100_thumb_index_6d",
    "hand_middle_ring": "sg100_middle_ring_5d",
}
RAD_TO_DEG = 180.0 / math.pi


def _processors(processor):
    entries = getattr(processor, "processors", None)
    return dict(entries) if entries is not None else {"rm65_sg100": processor}


def is_rm665(processor):
    entries = _processors(processor)
    return bool(entries) and all(
        set(getattr(getattr(p, "joint_bounds", None), "limits", {})) == set(GROUP_NAMES)
        for p in entries.values()
    )


def degree_log_metrics(metrics, processor):
    """Rename physical MAEs with explicit degree units; leave losses/ratios unchanged.

    Apply only at the tracker boundary, after existing distributed aggregation.
    Never reuse an old, unitless radian key for degree values.
    """
    if not is_rm665(processor):
        return dict(metrics)
    slot_groups = dict(GROUP_NAMES)
    for p in _processors(processor).values():
        for slot, alternatives in p.action_state_merger.merge_spec.items():
            physical = [key for key in alternatives if key in GROUP_NAMES]
            if len(physical) == 1:
                slot_groups[slot] = GROUP_NAMES[physical[0]]
    output = {}
    overall = {
        f"eval/action/rollout/{branch}{bounded}_action_l1":
        f"eval/action/rollout/{branch}_action_mae_deg_{'bounded' if bounded else 'unclipped'}"
        for branch in ("ar", "fm") for bounded in ("", "_bounded")
    }
    for key, value in metrics.items():
        grouped = re.fullmatch(
            r"eval/action/rollout/per_key/([^/]+)/l1/(ar|fm)_action_l1", key
        )
        per_emb = re.fullmatch(r"per_emb/([^/]+)/eval/(ar|fm)_action_l1", key)
        if key in overall:
            output[overall[key]] = float(value) * RAD_TO_DEG
        elif grouped and grouped[1] in slot_groups:
            name = f"eval/action/rollout/per_group/{slot_groups[grouped[1]]}/{grouped[2]}_action_mae_deg_unclipped"
            output[name] = float(value) * RAD_TO_DEG
        elif per_emb and per_emb[1] in _processors(processor):
            output[f"per_emb/{per_emb[1]}/eval/{per_emb[2]}_action_mae_deg_unclipped"] = float(value) * RAD_TO_DEG
        else:
            output[key] = value
    return output


def _error_statistics(errors, device):
    """Globally weighted MAE/max, not an average of per-rank means.

    Values are absolute errors in degrees with padding already excluded. A
    nonfinite prediction yields infinite MAE/max plus an explicit failure ratio.
    """
    names = list(errors)
    sums, maxima = [], []
    for name in names:
        values = errors[name].detach().double().reshape(-1)
        finite = torch.isfinite(values)
        sums.append([values[finite].sum().item(), values.numel(), (~finite).sum().item()])
        maxima.append(values[finite].max().item() if finite.any() else 0.0)
    totals = torch.tensor(sums, dtype=torch.float64, device=device)
    peaks = torch.tensor(maxima, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    result = {}
    for index, name in enumerate(names):
        total, count, nonfinite = totals[index].tolist()
        bad = nonfinite > 0
        mae = float("inf") if bad else (total / count if count else float("nan"))
        peak = float("inf") if bad else (peaks[index].item() if count else float("nan"))
        result[name] = dict(mae=mae, maximum=peak, count=int(count),
                            nonfinite_fraction=nonfinite / count if count else 0.0)
    return result


@torch.no_grad()
def codec_reconstruction_metrics(batch, model, processor):
    """True codec encode/decode on the current held-out batch, before limits.

    vs_codec_input isolates codec quantization after restoring physical units.
    vs_recorded_action includes any preprocessing/normalization representation
    loss, using the unmodified absolute-radian labels in gt_action.
    """
    if not is_rm665(processor):
        return {}
    frontend = model.action_tokenizer
    backend = frontend.action_tokenizer
    samples = batch.get("samples")
    if not samples or any("action" not in sample or "value" not in sample["action"] for sample in samples):
        raise ValueError("RM665 codec diagnostics require the actual samples[].action.value tokenizer inputs")
    inputs = torch.stack([sample["action"]["value"] for sample in samples])
    if inputs.ndim != 3 or inputs.shape[-1] != 27:
        raise ValueError("RM665 codec diagnostics require the official padded 27D layout")
    device = getattr(frontend, "device", inputs.device)
    inputs = inputs.to(device=device, dtype=torch.float32)
    pad = batch["action_dim_is_pad"].to(device=device, dtype=torch.bool)
    op_mask = batch.get("action_op_mask")
    if op_mask is not None:
        op_mask = op_mask.to(device=device)
    levels = getattr(frontend, "_default_num_residuals", None)
    if levels is None:
        levels = getattr(backend, "_default_num_residuals", None)
    if levels is None:
        levels = backend.model.config.n_codebooks
    levels = int(levels)
    # Prevent dropout/quantizer EMA updates without changing the original modes.
    modes = [(module, module.training) for module in backend.modules()]
    try:
        backend.eval()
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            codes = backend.encode(inputs, action_dim_is_pad=pad,
                                   action_op_mask=op_mask, num_residuals=levels)
            decoded = backend.decode(codes, action_dim_is_pad=pad,
                                     num_residuals=levels, target_time_steps=inputs.shape[1])
    finally:
        for module, training in modes:
            module.training = training
    if not isinstance(decoded, torch.Tensor) or decoded.shape != inputs.shape:
        raise ValueError("Codec reconstruction shape differs from its actual tokenizer input")
    embodiments = batch.get("embodiment")
    proprio = batch["proprio"].detach().cpu()
    restored = _denormalize_tokenizer_action_batch(decoded.detach().cpu(), proprio, processor, embodiments)
    restored_input = _denormalize_tokenizer_action_batch(inputs.detach().cpu(), proprio, processor, embodiments)
    recorded = batch.get("gt_action")
    if not isinstance(recorded, torch.Tensor) or recorded.shape != restored.shape:
        raise ValueError("RM665 codec diagnostics require original absolute-radian gt_action")
    recorded = recorded.detach().cpu()
    valid = (~pad.detach().cpu()).unsqueeze(1).expand_as(recorded).clone()
    time_pad = batch.get("action_is_pad")
    if isinstance(time_pad, torch.Tensor):
        time_pad = time_pad.detach().cpu().bool()
        if time_pad.ndim == 3 and time_pad.shape[-1] == 1:
            time_pad = time_pad.squeeze(-1)
        if time_pad.shape != recorded.shape[:2]:
            raise ValueError("Unexpected action_is_pad shape in codec diagnostics")
        valid &= ~time_pad.unsqueeze(-1)
    errors = {(display, reference): [] for display in GROUP_NAMES.values()
              for reference in ("recorded_action", "codec_input")}
    entries = _processors(processor)
    for index in range(len(samples)):
        p = entries[embodiments[index]] if embodiments is not None else next(iter(entries.values()))
        merger = p.action_state_merger
        prediction = merger.backward({"action": restored[index]})["action"]
        mask = merger.backward({"action": valid[index]})["action"]
        targets = {"recorded_action": merger.backward({"action": recorded[index]})["action"],
                   "codec_input": merger.backward({"action": restored_input[index]})["action"]}
        for (group, display), width in zip(GROUP_NAMES.items(), (6, 6, 5)):
            if prediction[group].shape[-1] != width:
                raise ValueError(f"Unexpected physical joint count for {group}")
            for reference, truth in targets.items():
                error = (prediction[group] - truth[group]).abs()[mask[group].bool()]
                errors[(display, reference)].append(error * RAD_TO_DEG)
    errors = {name: torch.cat(chunks) for name, chunks in errors.items()}
    output = {"eval/codec_encode_decode_unclipped/rvq_num_residuals": levels}
    for (display, reference), values in _error_statistics(errors, inputs.device).items():
        prefix = f"eval/codec_encode_decode_unclipped/{display}"
        output[f"{prefix}/mae_deg_vs_{reference}"] = values["mae"]
        output[f"{prefix}/max_abs_error_deg_vs_{reference}"] = values["maximum"]
        output[f"{prefix}/nonfinite_fraction_vs_{reference}"] = values["nonfinite_fraction"]
        output[f"{prefix}/valid_scalar_count_vs_{reference}"] = values["count"]
    return output
