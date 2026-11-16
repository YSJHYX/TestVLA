"""Pi0.5-compatible metric names, with local receipts and no login code."""
import json
import subprocess
import resource
import time
from pathlib import Path
import torch


class TrainingMetrics:
    def __init__(self, output_dir, train_frames, effective_batch_size, initial_step=0, initial_samples=None):
        self.root = Path(output_dir)
        self.frames = int(train_frames)
        self.batch = int(effective_batch_size)
        if self.frames <= 0 or self.batch <= 0:
            raise ValueError('Training metrics require nonempty data and positive batch size')
        self.last_step = initial_step
        self.samples_seen = int(initial_samples if initial_samples is not None else initial_step*self.batch)
        self.last_time = time.monotonic()
        self.loss_sum = 0.0
        self.loss_count = 0

    def manifest(self, limits=None):
        from g05.adapters.joint_bounds import default_limits
        repo = Path(__file__).resolve().parents[3]
        payload = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                   'mapping': 'arm6 + thumb_index6 + middle_ring5 -> original 27D',
                   'selected_train_frames': self.frames, 'effective_batch_size': self.batch,
                   'position_unit': 'SDK-user radians', 'physical_joint_limits': limits or default_limits()}
        (self.root/'run_manifest.json').write_text(json.dumps(payload,indent=2))
        return payload

    def observe_loss(self, loss, samples):
        self.loss_sum += float(loss)
        self.loss_count += 1
        self.samples_seen += int(samples)

    def log(self, step, learning_rate, grad_norm):
        now = time.monotonic()
        elapsed = now - self.last_time
        payload = dict(loss=self.loss_sum / max(1, self.loss_count),
                       learning_rate=float(learning_rate), grad_norm=float(grad_norm),
                       step=int(step), global_step=int(step),
                       time_per_step=elapsed / max(1, step-self.last_step),
                       samples_seen=self.samples_seen, equivalent_epochs=self.samples_seen/self.frames,
                       **{'system/cpu_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
        if torch.cuda.is_available():
            payload.update({'gpu/memory_allocated_bytes': torch.cuda.memory_allocated(),
                            'gpu/memory_reserved_bytes': torch.cuda.memory_reserved(),
                            'gpu/peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'gpu/peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        self.append('training_metrics.jsonl', payload)
        self.last_time, self.last_step = now, step
        self.loss_sum, self.loss_count = 0., 0
        return payload

    def append(self, filename, payload):
        text = json.dumps(payload, allow_nan=False, sort_keys=True)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / filename).open('a') as stream:
            stream.write(text+'\n')

    def checkpoint(self, step, model, optimizer=None):
        path = self.root / 'checkpoints' / f'step_{step}.pt'
        def tensor_bytes(values):
            if isinstance(values, torch.Tensor):
                return values.numel()*values.element_size()
            if isinstance(values, dict):
                return sum(tensor_bytes(v) for v in values.values())
            if isinstance(values, (list,tuple)):
                return sum(tensor_bytes(v) for v in values)
            return 0
        payload = {'checkpoint_step': int(step),
                   'checkpoint/total_bytes': path.stat().st_size,
                   'checkpoint/model_tensor_bytes': tensor_bytes(model.state_dict()),
                   'checkpoint/optimizer_tensor_bytes': tensor_bytes(optimizer.state) if optimizer else 0,
                   'checkpoint/save_complete': int(path.is_file() and path.stat().st_size>0),
                   'checkpoint/backup_verified': 0}
        # The native G0.5 artifact is one combined .pt file. These Pi0.5 aliases
        # report uncompressed tensor bytes, not nonexistent separate files.
        payload['checkpoint/model_bytes'] = payload['checkpoint/model_tensor_bytes']
        payload['checkpoint/optimizer_bytes'] = payload['checkpoint/optimizer_tensor_bytes']
        self.append('checkpoint_metrics.jsonl', payload)
        return payload


def log_camera_views(accelerator, batch, step):
    """Log the latest frame of each camera, once, on rank zero."""
    if not accelerator.is_main_process:
        return
    import numpy as np
    import wandb
    pixels = batch.get('pixel_values')
    if not isinstance(pixels, dict):
        return
    frames = []
    for name, value in pixels.items():
        if not isinstance(value, torch.Tensor):
            continue
        image = value.detach().float().cpu()
        while image.ndim > 3:
            image = image[0] if image.ndim > 4 else image[-1]
        frames.append(wandb.Image(((image+1)/2).clamp(0,1).permute(1,2,0).numpy(), caption=name))
    if frames:
        accelerator.log({'camera_views': frames}, step=step)
