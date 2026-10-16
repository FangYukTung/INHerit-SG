"""Run the current INHerit-SG retriever on an HM3DSem-SQR split."""
import argparse
import json
from pathlib import Path

from inherit_sg.cli import load_config
from inherit_sg.inference import query_components, retriever_for
from inherit_sg.query import parse_object


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/retrieval.yaml')
    p.add_argument('--maps-root',required=True,help='Contains <scene_id>/ep_000 maps')
    p.add_argument('--split',default='data/splits/main_864.json')
    p.add_argument('--output',required=True,help='Variant directory for prediction JSONL files')
    p.add_argument('--scene-id')
    p.add_argument('--no-visual',action='store_true')
    p.add_argument('--constraints',help='JSON object: split key -> validated recursive constraints')
    args=p.parse_args()
    config=load_config(args.config)
    rows=json.loads(Path(args.split).read_text())['queries']
    if args.scene_id: rows=[row for row in rows if row['scene_id']==args.scene_id]
    if not rows: raise ValueError('No queries selected')
    supplied=json.loads(Path(args.constraints).read_text()) if args.constraints else None
    if supplied is not None:
        supplied={row['key']:parse_object(supplied[row['key']]) for row in rows}
    scenes=sorted({row['scene_id'] for row in rows})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for scene in scenes:
        if (output/f'{scene}.jsonl').exists(): raise FileExistsError(output/f'{scene}.jsonl')
    parser,reranker=query_components(config,parse=supplied is None,visual=not args.no_visual)
    for scene in scenes:
        retriever=retriever_for(Path(args.maps_root)/scene/'ep_000',config)
        with (output/f'{scene}.jsonl').open('x') as handle:
            for row in rows:
                if row['scene_id']!=scene: continue
                text=row['query_text']
                parsed=supplied[row['key']] if supplied is not None else parser.parse(text)
                prediction=retriever.retrieve(text,parsed,reranker)
                record={key:row[key] for key in ('key','scene_id','query_id')}
                record.update(prediction)
                handle.write(json.dumps(record)+'\n');handle.flush()


if __name__=='__main__': main()
