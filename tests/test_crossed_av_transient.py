from types import SimpleNamespace as NS

import numpy as np
import pytest

from simulations.active_inference.experiments.crossed_av_transient import select_first, install_recorded
from simulations.active_inference.experiments.crossed_av_transient_analysis import analyze


def test_selection_is_first_threshold_crossing_not_best_score():
    rows=[dict(key='first',ticks=100),dict(key='best',ticks=100)]
    a=np.full((100,4,2),-.5);b=np.full((100,4,2),1.)
    a[20:22]=.002;a[10,0]=1.
    row,tick=select_first(rows,dict(first_bounds=a,best_bounds=b))
    assert row is rows[0] and tick==20
    with pytest.raises(ValueError,match='no same-episode'):
        select_first([dict(key='first',ticks=50)],dict(first_bounds=a))


def test_weight_transfer_rejects_reordered_sources_and_truncation():
    points={i:NS(u_i=NS(info=0.)) for i in [2,5]}
    node=NS(prediction_ports=[2,5],synapse_sources={2:(10,8),5:(20,8)},postsynaptic_points=points)
    net=NS(network=NS(neurons={1:node}))
    data=dict(prediction_ids=np.array([1]),context_source_ids=np.array([[10,20]]))
    install_recorded(net,data,np.array([[.2,.3]]))
    assert [points[i].u_i.info for i in [2,5]]==[.2,.3]
    with pytest.raises(ValueError,match='Invalid'):
        install_recorded(net,data,np.array([[.7]]))
    data['context_source_ids']=np.array([[20,10]])
    with pytest.raises(ValueError,match='identity'):
        install_recorded(net,data,np.array([[.2,.3]]))


def test_empty_transient_analysis_cannot_be_a_success(tmp_path):
    with pytest.raises(ValueError,match='No transient evidence'):
        analyze([],tmp_path/'empty')
    assert not (tmp_path/'empty').exists()
