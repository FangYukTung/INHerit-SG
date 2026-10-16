"""Serialize current memberships after incremental hierarchy refreshes."""
import csv
import json
from pathlib import Path


def normalize_hierarchy(directory, valid_rooms):
    """Discard superseded areas/rooms; retain one current parent per child.

    Room snapshots name their current areas. Object memory names its current
    room/area. Historical append-only relations are not a second authority.
    """
    directory=Path(directory)
    memory=json.loads((directory/'memory_kb.json').read_text())
    nodes={}
    for line in (directory/'nodes.jsonl').read_text().splitlines():
        if line.strip():
            node=json.loads(line);nodes[node['node_id']]=node
    room_ids={f'room_{rid}' for rid in valid_rooms}
    rooms={key:value for key,value in nodes.items() if value.get('node_level')==1 and key in room_ids}
    areas={}
    for room_id,room in rooms.items():
        for area_id in room.get('contained_area_ids',[]):
            if area_id in nodes and nodes[area_id].get('parent_room_id')==room_id:
                areas[area_id]=nodes[area_id]
    for area in areas.values(): area['contained_obj_nodes']=[]
    for obj in memory:
        area_id=obj.get('parent_area');room_id=f"room_{obj.get('parent_room')}"
        if area_id not in areas or areas[area_id]['parent_room_id']!=room_id:
            obj['parent_area']=None
            if room_id not in rooms: obj['parent_room']=None
        else:
            areas[area_id]['contained_obj_nodes'].append(obj['node_id'])
    areas={key:area for key,area in areas.items() if area['contained_obj_nodes']}
    for room in rooms.values():
        room['contained_area_ids']=[key for key in room.get('contained_area_ids',[]) if key in areas]
    floors={key:value for key,value in nodes.items() if value.get('node_level')==0}
    floor_by_index={value['floor_index']:key for key,value in floors.items()}
    for floor in floors.values(): floor['contained_room_nodes']=[]
    edges=[]
    for room_id,room in rooms.items():
        floor_id=floor_by_index.get(room.get('floor_id'))
        if floor_id:
            edges.append((room_id,floor_id,1));floors[floor_id]['contained_room_nodes'].append(room_id)
    for area_id,area in areas.items(): edges.append((area_id,area['parent_room_id'],2))
    objects={}
    for obj in memory:
        objects[obj['node_id']]={**nodes.get(obj['node_id'],{}),**obj,'node_level':3}
        if obj.get('parent_area'): edges.append((obj['node_id'],obj['parent_area'],3))
    ordered={**floors,**rooms,**areas,**objects}
    with (directory/'nodes.jsonl').open('w') as handle:
        for node in ordered.values(): handle.write(json.dumps(node)+'\n')
    with (directory/'relations.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['child_node_id','parent_node_id','node_level']);writer.writerows(edges)
    (directory/'memory_kb.json').write_text(json.dumps(memory,indent=2)+'\n')
