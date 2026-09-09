"""Prepare explicit terminal architectures for the existing media-order runner.

This creates a configuration input, not a fabricated training recording. The
existing runner takes the prepared config and the unchanged physical feature
paths; it performs fresh acquisition and all standard order-controlled probes.
"""
import argparse
import inspect
import json
from pathlib import Path

from ..components.learning.projection_terminals import contact_terminals, mean_pooled_return_rates
from .association_route_probe import digest
from .composition_probe import encode


def prepare(source, output, mode):
    if mode not in ('contact','mean_pooled'):raise ValueError('Unknown terminal architecture')
    source,output=Path(source).resolve(),Path(output).resolve()
    old=json.loads((source/'manifest.json').read_text())
    if old['condition']!='eligibility':raise ValueError('Requires the existing eligibility graph')
    if any(digest(p)!=h for p,h in old['source_hashes'].items()):raise ValueError('Source runtime changed')
    cfg=json.loads((source/'config.json').read_text())
    builder=contact_terminals if mode=='contact' else mean_pooled_return_rates
    cfg,report=builder(cfg,source_ids=old['groups']['visual_core'])
    hashes=dict(old['source_hashes'])
    for obj in (prepare,builder):
        path=Path(inspect.getfile(obj)).resolve();hashes[str(path)]=digest(path)
    m={**old,'source_hashes':hashes,'prepared_from':str(source),
       'prepared_source_files':{str(source/name):digest(source/name) for name in ('manifest.json','config.json')},
       'configuration_intervention':dict(mode=mode,report=report),
       'preparation_status':'Configuration input only. No acquisition or probe result is claimed by this manifest.'}
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(encode(cfg)+'\n');(output/'manifest.json').write_text(encode(m)+'\n')
    return dict(output=str(output),mode=mode,neurons=len(cfg['neurons']),connections=len(cfg['connections']),
                terminals=sum(p['type']=='presynaptic' for p in cfg['synaptic_points']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('contact','mean_pooled'),required=True)
    a=p.parse_args();print(encode(prepare(a.source,a.output,a.mode)))
