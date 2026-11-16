"""CPU data-path check through the actual task, reader, stats and processor."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import torch
from hydra import initialize_config_dir, compose
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.data.data_utils import collate_fn_pad_sequences


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True,help='JSON validation receipt')
    parser.add_argument('--workers',type=int,default=0,help='DataLoader workers for this check')
    args=parser.parse_args()
    output_path=Path(args.output).resolve()
    torch.set_num_threads(4)
    os.environ['G05_DATASET_DIR']=str(Path(args.dataset).resolve())
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=rm65_sg100_665','data.val_set_proportion=0.0'])
    processors=build_processors(cfg)
    dataset=instantiate_dataset(cfg,is_training_set=True)
    stats=dataset.get_dataset_stats(processors)
    processors.set_normalizer_from_stats(stats)
    dataset.set_processor(processors)
    loader=torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset,[0,min(150,len(dataset)-1)]),
        batch_size=2,num_workers=args.workers,collate_fn=collate_fn_pad_sequences)
    batch=next(iter(loader))
    assert batch['action'].shape==(2,32,27)
    assert batch['proprio'].shape==(2,6,27)
    assert all(v.shape==(2,6,3,256,256) for v in batch['pixel_values'].values())
    # Native dummy is a black RGB image; Normalize(mean=.5,std=.5) maps it to -1.
    assert torch.all(batch['pixel_values']['wrist_right']==-1)
    proc=processors['rm65_sg100']
    out=proc.postprocess({'action_ar':batch['action'],'action_fm':batch['action'], 'proprio':batch['proprio']})
    assert out['action_17d'].shape==(2,32,17)
    assert torch.equal(out['action_fm_17d'],out['action_ar_17d'])
    receipt={'reader':'original BaseLerobotDataset / v2.1','processor':type(proc).__name__,
             'frames':len(dataset),'action_shape':list(batch['action'].shape),
             'state_history_shape':list(batch['proprio'].shape),
             'rgb_shapes':{k:list(v.shape) for k,v in batch['pixel_values'].items()},
             'right_wrist_dummy_black':True,'physical_output_shape':list(out['action_17d'].shape),
             'both_output_paths_tested':True,'history_did_not_trim_action_horizon':True,
             'dataloader_workers':args.workers,
             'boundary_metrics':out['boundary_metrics'],'extra_cot_labels_required':False,
             'gpu_policy_training_tested':False}
    output_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
