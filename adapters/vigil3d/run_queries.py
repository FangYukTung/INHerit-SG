"""Ground query-only JSON to fixed proposal boxes using the current pipeline.

Input: [{"scene_id": "...", "ann_id": "...", "query_text": "..."}].
No ground-truth target fields enter the parser, retrieval, or set selector.
"""
import argparse
import json
from pathlib import Path

from inherit_sg.cli import load_config
from inherit_sg.inference import query_components, retriever_for
from inherit_sg.model_router import ModelRouter
from mask3d_structural_adapter import latest_native_map
from set_grounding import select_set


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries',required=True)
    p.add_argument('--maps-root',required=True)
    p.add_argument('--config',default='configs/retrieval.yaml')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    rows=json.loads(Path(args.queries).read_text())
    keys=[f"{row['scene_id']}_{row['ann_id']}" for row in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate annotation IDs')
    config=load_config(args.config)
    parser,reranker=query_components(config)
    routing=dict(config['llm_config']['model_router'])
    routing['routing']={'set_selection':routing['routing']['visual_reranking']}
    router=ModelRouter(routing)
    client,model=router.resolve('set_selection')
    predictions={};traces=[]
    for scene in sorted({row['scene_id'] for row in rows}):
        retriever=retriever_for(latest_native_map(Path(args.maps_root),scene),config)
        for row in rows:
            if row['scene_id']!=scene: continue
            text=row['query_text'];key=f"{scene}_{row['ann_id']}"
            result=retriever.retrieve(text,parser.parse(text),reranker)
            selected,trace=select_set(client,model,text,result['candidates'],retriever.memory)
            predictions[key]=[retriever.memory.objects[node_id]['aabb_3d'] for node_id in selected]
            traces.append({'key':key,'selected':selected,**trace})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle: json.dump(predictions,handle,indent=2)
    with output.with_suffix('.trace.json').open('x') as handle: json.dump(traces,handle,indent=2)


if __name__=='__main__': main()
