"""G0.5 processor extension; network, codec, CoT and visual memory unchanged."""
from .galaxea_cot_processor import GalaxeaCoTProcessor
from g05.adapters.joint_bounds import JointBounds, physical_action


class RM65SG100Processor(GalaxeaCoTProcessor):
    def __init__(self, joint_limits=None, **kwargs):
        super().__init__(**kwargs)
        self.joint_bounds = JointBounds(joint_limits)
        expected = dict(left_control=9,left_gripper=1,right_control=9,right_gripper=1,lower_body=7)
        if dict(self.action_state_merger.max_action_shape_meta) != expected:
            raise ValueError('6+6+5 adapter must retain the original 9+1+9+1+7 layout')

    def set_normalizer_from_stats(self, dataset_stats=None):
        for domain,prefix in [('action','stepwise_'),('state','global_')]:
            for key,width in zip(('arm','hand_thumb_index','hand_middle_ring'),(6,6,5)):
                fields = (dataset_stats or {}).get(domain,{}).get(key,{})
                if not fields or any(getattr(v,'shape',())[-1:] != (width,) for name,v in fields.items() if name.startswith(prefix)):
                    raise ValueError(f'Missing or incompatible task statistics: {domain}/{key}')
        return super().set_normalizer_from_stats(dataset_stats)

    def postprocess(self, data):
        # This dataset's 32 actions start at t=0, irrespective of its six history
        # frames. Do not inherit the generic processor's history-based trimming.
        output = dict(data)
        source_state = data.get('state', data.get('proprio'))
        if source_state is None:
            raise ValueError('Normalized proprio/state is required to restore RM65 actions')
        paths = [k for k in ('action_fm', 'action_ar') if k in data]
        if not paths:
            paths = ['action']
        metrics = {}
        for key in paths:
            unmerged = self.action_state_merger.backward(
                {'action': data[key], 'state': source_state})
            action, state = self.joint_bounds.decode(unmerged['action'], unmerged['state'], self.normalizer)
            output[key] = action
            output[key + '_17d'] = physical_action(action)
            metrics.update({f'{key}/{k}': v for k, v in self.joint_bounds.last_metrics.items()})
        if paths != ['action']:
            chosen = 'action_fm' if 'action_fm' in output else 'action_ar'
            output['action'] = output[chosen]
            output['action_17d'] = output[chosen + '_17d']
        output['boundary_metrics'] = metrics
        return output
