"""Independent added-path, existing-learning and body/resource tick checks."""
import heapq
import math

import numpy as np

from . import context_organization as base
from .active_sweep_transfer import audit as audit_predictor
from .ventilation_screen import audit as audit_body


def audit(z,cfg,groups,features):
    audit_body(z)
    audit_predictor(z,cfg,groups,features,.8)
    meta=cfg['metadata']['ventilation_feedback'];ids=list(z['neuron_ids']);observed=list(z['reg_ids'])
    expected_ids=meta['neurons']+groups['muscle']
    if observed!=expected_ids: raise ValueError('Wrong observed population')
    nodes={n['id']:n for n in cfg['neurons']};index={n:ids.index(n) for n in observed}
    n=len(z['body']);width=z['reg_q_before'].shape[-1]
    if (n<1 or z['reg_inputs'].shape!=(n,len(observed),width,4)
            or z['reg_returns'].ndim!=2 or z['reg_returns'].shape[1]!=8
            or z['reg_return_offsets'].shape!=(n+1,)
            or z['reg_return_offsets'][0]!=0 or z['reg_return_offsets'][-1]!=len(z['reg_returns'])
            or np.any(np.diff(z['reg_return_offsets'])<0)):
        raise ValueError('Missing full neural evidence')
    queue={nid:[] for nid in observed}
    for nid,due,sid,v in z['reg_queues_initial']:
        heapq.heappush(queue[int(nid)],(int(due),'hillock',np.float32(v),int(sid)))
    points={(p['neuron_id'],p['synapse_id']):p for p in cfg['synaptic_points'] if p['type']=='postsynaptic'}
    edges={(e['target_neuron'],e['target_synapse']):(e['source_neuron'],e['source_terminal']) for e in cfg['connections']}
    terms=list(map(tuple,z['terminal_ids']))
    cpg=list(z['cpg_ids'])
    if cpg!=groups['cpg'] or z['cpg_inputs'].shape!=(n,len(cpg),2,4):
        raise ValueError('Missing actual rhythm input evidence')
    old_cells=z['reg_cells_initial'];old_terms=z['terminal_initial']
    due_sensory=list(z['organ_delay_initial'].copy())
    np.testing.assert_array_equal(due_sensory,np.tile([.75,.25,.5],(64,1)))
    before_organ=z['organ_initial'];qprevious=None;max_residual=0.
    start=round(z['body'][0,0]/.004)-1
    for t in range(n):
        expected_cpg=z['cpg_arriving_initial'].copy() if t==0 else np.zeros((len(cpg),2,4),dtype=np.float32)
        if t>0:
            for j,nid in enumerate(cpg):
                for sid in range(2):
                    if (nid,sid) in edges:
                        src,term=edges[nid,sid]
                        expected_cpg[j,sid,0]=old_cells[ids.index(src),base.FIELDS.index('O')]*old_terms[terms.index((src,term))]
        # Birth injection and the recurrent edge share this port. The runtime
        # scatters external drive first, then adds arriving neural releases.
        expected_cpg[0,0,0]+=5. if start+t==0 else 0.
        np.testing.assert_array_equal(z['cpg_inputs'][t],expected_cpg)
        np.testing.assert_array_equal(z['organ_before'][t],before_organ)
        raw=np.array([before_organ[6]/.3,1-before_organ[6]/.3,before_organ[0]/.3])
        np.testing.assert_array_equal(z['organ_raw'][t],raw)
        due_sensory.append(raw);drive=due_sensory.pop(0)
        np.testing.assert_array_equal(z['organ_drive'][t],drive)
        expected=z['reg_arriving_initial'].copy() if t==0 else np.zeros_like(z['reg_inputs'][t])
        if t>0:
            for j,nid in enumerate(observed):
                for sid in range(nodes[nid]['params']['num_inputs']):
                    if (nid,sid) in edges:
                        src,term=edges[nid,sid]
                        expected[j,sid,0]=old_cells[ids.index(src),base.FIELDS.index('O')]*old_terms[terms.index((src,term))]
        for nid,value in zip(meta['sensory_ids'],drive): expected[observed.index(nid),0,0]=value
        np.testing.assert_array_equal(z['reg_inputs'][t],expected)
        if qprevious is not None: np.testing.assert_array_equal(z['reg_q_before'][t],qprevious)
        scheduled=np.zeros((len(observed),width),dtype=np.float32)
        qnext=z['reg_q_before'][t].copy();returns=[]
        # Network iteration order matters to the ordered return record.
        for nid in sorted(observed,key=ids.index):
            j=observed.index(nid);node=nodes[nid];count=node['params']['num_inputs']
            if node['metadata'].get('graded_gain',0)<=0 or node['params']['eta_post']<=0:
                raise ValueError('Observer assumes continuously adaptive graded added path')
            for sid in range(count):
                a=expected[j,sid];q=float(z['reg_q_before'][t,j,sid]);p=points[nid,sid]
                if p['u_i']['plast']!=0: raise ValueError('Unexpected plastic throughput')
                if a[0]>0:
                    scheduled[j,sid]=a[0]*q
                    heapq.heappush(queue[nid],(start+t+p['distance_to_hillock'],'hillock',scheduled[j,sid],sid))
                    ev=np.array([a[0]-q,a[1],*a[2:]],dtype=np.float32)
                    qnext[j,sid]=math.copysign(abs(q)*math.exp(-node['params']['eta_post']*(float(np.linalg.norm(ev))+.02)),q)
                    if (nid,sid) in edges:
                        src,term=edges[nid,sid]
                        if q<0 and node['metadata'].get('retrograde_magnitude_error',False): ev[0]=a[0]-abs(q)
                        returns.append([nid,sid,src,term,*(-ev)])
            current=0.
            while queue[nid] and queue[nid][0][0]<=start+t:
                _,_,v,sid=heapq.heappop(queue[nid])
                current += v*(node['params']['delta_decay']**points[nid,sid]['distance_to_hillock'])
            prior=np.float32(old_cells[index[nid],base.FIELDS.index('S')])
            s=prior+(1./node['params']['lambda_param'])*(-prior+current)
            output=node['metadata']['graded_gain']*max(0.,float(s)-node['metadata'].get('graded_S0',0.))
            actual=z['cells'][t,index[nid],[base.FIELDS.index('S'),base.FIELDS.index('O')]]
            residual=float(np.max(abs(actual-[s,output])));max_residual=max(max_residual,residual)
            np.testing.assert_allclose(actual,[s,output],rtol=0,atol=3e-6)
        np.testing.assert_array_equal(z['reg_scheduled'][t],scheduled)
        np.testing.assert_allclose(z['reg_q_after'][t],qnext,rtol=0,atol=2e-12)
        actual_returns=z['reg_returns'][z['reg_return_offsets'][t]:z['reg_return_offsets'][t+1]]
        np.testing.assert_array_equal(actual_returns,np.asarray(returns).reshape(-1,8))
        qprevious=z['reg_q_after'][t];old_cells=z['cells'][t];old_terms=z['terminal_info'][t]
        before_organ=z['organs'][t]
    np.testing.assert_array_equal(due_sensory,z['organ_delay_final'])
    return max_residual
