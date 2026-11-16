import math
from types import SimpleNamespace
import pytest
import torch
from g05.adapters.joint_bounds import JointBounds, PARTS, default_limits, physical_action
from g05.utils.data.normalizer import SingleFieldLinearNormalizer
from g05.adapters.training_metrics import TrainingMetrics
from pathlib import Path
from copy import deepcopy


@pytest.fixture
def processor(monkeypatch):
    from hydra import initialize_config_dir,compose
    from g05.utils.data.processor_utils import build_processors
    root=Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv('G05_DATASET_DIR','unused-for-config-test')
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=rm65_sg100_665'])
    p=build_processors(cfg)['rm65_sg100']
    p._normalizer=normalizers('z-score')
    return p


def normalizers(mode):
    limits = default_limits()
    ns = {'action': {}, 'state': {}}
    for key, width in zip(PARTS, (6,6,5)):
        lo = torch.tensor(limits[key]['lower'])
        hi = torch.tensor(limits[key]['upper'])
        mean = (lo+hi)/2
        for domain in ns:
            m = torch.zeros(width) if key == 'arm' and domain == 'action' else mean
            stats = dict(mean=m.clone(),std=torch.full((width,),.2),
                         q01=m-.1,q99=m+.1,min=lo.clone(),max=hi.clone())
            ns[domain][key] = SingleFieldLinearNormalizer(stats,mode=mode)
    return SimpleNamespace(normalizers=ns)


@pytest.mark.parametrize('mode',['z-score-tail','z-score','q01/q99'])
def test_extreme_outputs_bounded_before_inverse(mode):
    n = normalizers(mode);bounds=JointBounds()
    actions, states = {}, {}
    for key,width in zip(PARTS,(6,6,5)):
        a=torch.full((2,32,width),float('inf'));a[0]=-float('inf')
        s=torch.zeros((2,6,width))
        actions[key]=a;states[key]=n.normalizers['state'][key].forward(s)
    result,_=bounds.decode(actions,states,n)
    for key in PARTS:
        assert torch.isfinite(result[key]).all()
        lo=torch.tensor(bounds.limits[key]['lower']);hi=torch.tensor(bounds.limits[key]['upper'])
        assert (result[key]>=lo).all() and (result[key]<=hi).all()
        torch.testing.assert_close(result[key][0,0],lo,atol=1e-5,rtol=0)
        torch.testing.assert_close(result[key][1,0],hi,atol=1e-5,rtol=0)


def test_relative_arm_uses_latest_measured_state():
    n=normalizers('z-score');bounds=JointBounds();actions,states={},{}
    for key,width in zip(PARTS,(6,6,5)):
        s=torch.zeros(1,6,width)
        if key=='arm': s[:,-1]=1.0
        states[key]=n.normalizers['state'][key].forward(s)
        target=torch.zeros(1,32,width)
        if key=='arm':target[:]=.2
        actions[key]=n.normalizers['action'][key].forward(target)
    result,_=bounds.decode(actions,states,n)
    torch.testing.assert_close(result['arm'],torch.full((1,32,6),1.2))
    assert physical_action(result).shape==(1,32,17)


def test_nan_is_not_replaced_with_an_arbitrary_command():
    n=normalizers('z-score');a={k:torch.zeros(1,32,w) for k,w in zip(PARTS,(6,6,5))}
    s={k:torch.zeros(1,6,w) for k,w in zip(PARTS,(6,6,5))}
    a['arm'][0,0,0]=math.nan
    with pytest.raises(ValueError,match='NaN'):
        JointBounds().decode(a,s,n)


def test_omitted_ar_group_holds_measured_pose():
    n=normalizers('z-score-tail')
    a={k:torch.full((1,32,w),-100.) for k,w in zip(PARTS,(6,6,5))}
    s={k:n.normalizers['state'][k].forward(torch.full((1,6,w),.05)) for k,w in zip(PARTS,(6,6,5))}
    result,_=JointBounds().decode(a,s,n)
    # Hand J2/J4 have upper limit zero; measured values beyond limits are bounded.
    for k in PARTS:
        lo=torch.tensor(default_limits()[k]['lower']);hi=torch.tensor(default_limits()[k]['upper'])
        expected=torch.full_like(result[k],.05).clamp(min=lo,max=hi)
        torch.testing.assert_close(result[k],expected,atol=1e-5,rtol=0)


def test_metrics_match_pi05_exposure_and_local_receipt(tmp_path):
    m=TrainingMetrics(tmp_path,train_frames=100,effective_batch_size=8)
    m.observe_loss(1,16);m.observe_loss(3,24)
    payload=m.log(5,1e-4,2.)
    assert payload['loss']==2 and payload['samples_seen']==40
    assert payload['equivalent_epochs']==.4
    assert (tmp_path/'training_metrics.jsonl').read_text().count('\n')==1


def test_dual_path_history_preserves_32_actions(processor):
    parts={k:torch.zeros(32,w) for k,w in zip(PARTS,(6,6,5))}
    states={k:torch.zeros(6,w) for k,w in zip(PARTS,(6,6,5))}
    packed=processor.action_state_merger.forward({'action':parts,'state':states})
    assert packed['action'].shape==(32,27)
    data={'action_fm':packed['action'].unsqueeze(0),
          'action_ar':packed['action'].unsqueeze(0),'proprio':packed['state'].unsqueeze(0)}
    before=deepcopy(data)
    out=processor.postprocess(data)
    assert out['action_17d'].shape==(1,32,17)
    torch.testing.assert_close(out['action_ar_17d'],out['action_fm_17d'])
    for k in before:torch.testing.assert_close(data[k],before[k])


def test_checkpoint_receipt_and_resume_sample_count(tmp_path):
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
    (tmp_path/'checkpoints').mkdir()
    torch.save({'model_state_dict':model.state_dict()},tmp_path/'checkpoints/step_5.pt')
    m=TrainingMetrics(tmp_path,100,8,initial_step=5,initial_samples=37)
    m.observe_loss(1.,3)
    assert m.log(6,1e-4,1.)['samples_seen']==40
    receipt=m.checkpoint(5,model,optimizer)
    assert receipt['checkpoint/save_complete']==1
    assert receipt['checkpoint/backup_verified']==0


def test_eval_helpers_restore_relative_arm_without_clipping_labels(processor):
    from utils.metric import _denormalize_action_batch,_denormalize_tokenizer_action_batch
    # Helpers use the real LinearNormalizer interface; wrap these per-field norms.
    class Norm:
        def backward(self,batch):
            for domain in ('action','state'):
                for k,v in batch[domain].items():
                    batch[domain][k]=processor._normalizer.normalizers[domain][k].backward(v)
            return batch
    original=processor._normalizer
    wrapped=Norm()
    wrapped.normalizers=original.normalizers
    processor._normalizer=wrapped
    parts={k:torch.zeros(32,w) for k,w in zip(PARTS,(6,6,5))}
    states={k:original.normalizers['state'][k].forward(torch.zeros(6,w)) for k,w in zip(PARTS,(6,6,5))}
    states['arm']=original.normalizers['state']['arm'].forward(torch.ones(6,6))
    parts['arm']=original.normalizers['action']['arm'].forward(torch.full((32,6),.2))
    pack=processor.action_state_merger.forward({'action':parts,'state':states})
    for helper in (_denormalize_action_batch,_denormalize_tokenizer_action_batch):
        result=helper(pack['action'].unsqueeze(0),pack['state'].unsqueeze(0),processor=processor)
        # right_control occupies model slots 10:19; first six are the arm.
        torch.testing.assert_close(result[0,:,10:16],torch.full((32,6),1.2))


def test_bounded_eval_uses_same_output_limits(processor):
    from utils.metric import _bounded_action_batch
    prediction=torch.full((2,32,27),float('inf'))
    state=torch.zeros(2,6,27)
    bounded=_bounded_action_batch(prediction,state,processor)
    assert bounded.shape==(2,32,27)
    assert torch.isfinite(bounded).all()
    for start,width,key in [(0,6,'hand_thumb_index'),(10,6,'arm'),(20,5,'hand_middle_ring')]:
        upper=torch.tensor(default_limits()[key]['upper']).expand(2,32,width)
        torch.testing.assert_close(bounded[:,:,start:start+width],upper,atol=1e-5,rtol=0)
