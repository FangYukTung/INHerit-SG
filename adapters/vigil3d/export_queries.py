"""Export only instruction text and IDs from official ViGiL3D annotations."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import pandas as pd
    frame=pd.read_parquet(args.annotations,columns=['scene_id','ann_id','description'])
    rows=[{'scene_id':str(row.scene_id),'ann_id':str(row.ann_id),'query_text':str(row.description)}
          for row in frame.itertuples(index=False)]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle: json.dump(rows,handle,indent=2)


if __name__=='__main__': main()
