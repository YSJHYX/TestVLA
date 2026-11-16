"""Physical-domain boundaries for the RM65 + SG100 6+6+5 adapter.

Only inference/reconstruction outputs are clipped. Recorded labels and measured
states are never silently edited. All values are SDK-user radians.
"""
import math
import torch

PARTS = ("arm", "hand_thumb_index", "hand_middle_ring")
# Existing RM65 real backend imports these values from rm65_sim.py.
ARM_LOWER_RAD = [-3.106, -2.2689, -2.356, -3.106, -2.234, -6.28]
HAND_LOWER_DEG = [0, -150, -60, -90, 0, -60, 0, -60, 0, 0, -60]
HAND_UPPER_DEG = [125, 0, 90, 0, 125, 90, 125, 90, 180, 125, 90]


def default_limits():
    lower = ARM_LOWER_RAD + [math.radians(x) for x in HAND_LOWER_DEG]
    upper = [-x for x in ARM_LOWER_RAD] + [math.radians(x) for x in HAND_UPPER_DEG]
    return {k: {"lower": lower[i:j], "upper": upper[i:j]}
            for k, i, j in zip(PARTS, (0, 6, 12), (6, 12, 17))}


def physical_action(action):
    """Restore the original 17-channel order, without renaming dataset columns."""
    return torch.cat([action[k] for k in PARTS], dim=-1)


class JointBounds:
    def __init__(self, limits=None):
        self.limits = limits if limits is not None else default_limits()
        for k, width in zip(PARTS, (6, 6, 5)):
            lo = torch.tensor(self.limits[k]["lower"])
            hi = torch.tensor(self.limits[k]["upper"])
            if lo.shape != (width,) or hi.shape != lo.shape:
                raise ValueError(f"Invalid joint limit shape for {k}")
            if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and (lo < hi).all()):
                raise ValueError(f"Invalid joint limits for {k}")
        self.last_metrics = {}

    @staticmethod
    def _mapped_bound(normalizer, value):
        # Unlike normalizer.forward, this must NOT clip the transformed bounds
        # to [-5,5]: those clips are a training representation policy, not limits.
        if normalizer.mode == "dummy":
            return value
        if normalizer._tail_c_pos is not None:
            value = normalizer._apply_tail_forward(value)
        scale, offset = normalizer._match_horizon(value)
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Joint bounds require a finite, monotone normalizer")
        value = value * scale.to(value) + offset.to(value)
        if normalizer.mode == "tanh":
            value = value.tanh()
        return value

    def decode(self, actions, normalized_states, normalizer):
        """Inverse normalize, restore relative arm, and return physical parts.

        Input dictionaries are already unmerged, shaped [..., H, physical_D].
        normalized_states retain the whole observation history. Latest measured
        arm state defines both the relative limits and the inverse transform.
        """
        result, state = {}, {}
        metrics = {}
        for key in PARTS:
            a = actions[key].float()
            s = normalized_states[key].float()
            if torch.isnan(a).any() or not torch.isfinite(s).all():
                raise ValueError(f"Nonfinite observation or NaN action in {key}")
            # Official grouped AR decoder fills an omitted whole group with -100.
            # An absent group means hold, not move to its lower physical limit.
            absent = (a == -100.).all(dim=-1,keepdim=True).all(dim=-2,keepdim=True)
            sn = normalizer.normalizers['state'][key]
            state[key] = sn.backward(s)
            if not torch.isfinite(state[key]).all():
                raise ValueError(f"Nonfinite denormalized state in {key}")
            lo = a.new_tensor(self.limits[key]['lower']).expand_as(a)
            hi = a.new_tensor(self.limits[key]['upper']).expand_as(a)
            reference = state[key][..., -1:, :] if key == 'arm' else torch.zeros_like(lo)
            an = normalizer.normalizers['action'][key]
            mapped_lo = self._mapped_bound(an, lo - reference)
            mapped_hi = self._mapped_bound(an, hi - reference)
            if not (torch.isfinite(mapped_lo).all() and torch.isfinite(mapped_hi).all()):
                raise ValueError(f"Nonfinite mapped physical limits in {key}")
            bounded = torch.minimum(torch.maximum(a, mapped_lo), mapped_hi)
            raw = an.backward(bounded) + reference
            if not torch.isfinite(raw).all():
                raise ValueError(f"Inverse normalization remained nonfinite in {key}")
            final = torch.minimum(torch.maximum(raw, lo), hi)
            hold = torch.minimum(torch.maximum(state[key][..., -1:, :].expand_as(a),lo),hi)
            final = torch.where(absent,hold,final)
            metrics[f'boundary/{key}/pre_clip_fraction'] = float((bounded != a).float().mean())
            metrics[f'boundary/{key}/post_clip_fraction'] = float((final != raw).float().mean())
            metrics[f'boundary/{key}/held_absent_fraction'] = float(absent.float().mean())
            result[key] = final
        self.last_metrics = metrics
        return result, state
