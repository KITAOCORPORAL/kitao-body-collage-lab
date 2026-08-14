"""Anatomical body-part splitting from person mask + pose + SAM2."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .model_backends import Sam2Backend
from .utils.image_io import comfy_image_to_pil, pil_to_comfy_image
from .utils.mask_utils import masks_to_batch
from .utils.path_config import get_path_config
from .utils.pose_utils import corridor, ellipse_mask, part_record, point, polygon_mask
from .utils.preview_utils import render_mask_contact_sheet

STANDARD_PARTS = ["head", "torso", "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm", "right_forearm", "right_hand", "left_thigh", "left_calf", "left_foot", "right_thigh", "right_calf", "right_foot"]
PART_COLORS = [(239,71,111),(255,209,102),(6,214,160),(17,138,178),(7,59,76),(249,65,68),(144,190,109),(244,162,97),(87,117,144),(67,170,139),(249,199,79),(248,150,30),(87,204,153),(39,125,161)]


def _confidence(pose, names):
    values = [pose["keypoints"][name]["confidence"] for name in names if name in pose.get("keypoints", {})]
    return float(min(values)) if values else 0.0


def _cloud(points, threshold):
    return np.array([[p["x"], p["y"]] for p in points if p["confidence"] >= threshold], dtype=np.float32)


def _candidate_parts(pose, person_mask, threshold, include_head, include_hands, include_feet):
    shape = person_mask.shape
    parts, missing = [], []
    person_width = max(1, pose["bbox"][2] - pose["bbox"][0])
    def limb(label, a_name, b_name, scale):
        a, b = point(pose, a_name, threshold), point(pose, b_name, threshold)
        if a is None or b is None:
            missing.append(label); return
        width = max(6, np.linalg.norm(b - a) * scale, person_width * 0.045)
        mask = corridor(shape, a, b, width) & person_mask
        parts.append(part_record(label, mask, _confidence(pose, [a_name,b_name]), "person_01", a, a, b, "ok"))
    if include_head:
        face = _cloud(pose.get("face_points", []), threshold)
        nose = point(pose, "nose", threshold); ls=point(pose,"left_shoulder",threshold); rs=point(pose,"right_shoulder",threshold)
        if len(face) >= 8 and nose is not None:
            center = face.mean(axis=0) + np.array([0, -np.ptp(face,axis=0)[1] * 0.18]); extent=np.ptp(face,axis=0); axes=(max(12,extent[0]*0.95),max(16,extent[1]*1.18))
            mask=ellipse_mask(shape,center,axes)&person_mask
            parts.append(part_record("head",mask,float(np.mean([p["confidence"] for p in pose["face_points"]])),"person_01",center,quality="ok"))
        elif nose is not None and ls is not None and rs is not None:
            shoulder=np.linalg.norm(ls-rs); center=nose+np.array([0,-shoulder*0.12]); mask=ellipse_mask(shape,center,(shoulder*0.42,shoulder*0.52))&person_mask
            parts.append(part_record("head",mask,_confidence(pose,["nose","left_shoulder","right_shoulder"]),"person_01",center,quality="uncertain"))
        else: missing.append("head")
    torso_names=["left_shoulder","right_shoulder","left_hip","right_hip"]
    torso_pts=[point(pose,n,threshold) for n in torso_names]
    if all(v is not None for v in torso_pts):
        expand=max(4,int(np.linalg.norm(torso_pts[0]-torso_pts[1])*0.12)); mask=polygon_mask(shape,torso_pts,expand)&person_mask
        parts.append(part_record("torso",mask,_confidence(pose,torso_names),"person_01",np.mean(torso_pts,axis=0),quality="ok"))
    else: missing.append("torso")
    for side in ("left","right"):
        limb(f"{side}_upper_arm",f"{side}_shoulder",f"{side}_elbow",0.34)
        limb(f"{side}_forearm",f"{side}_elbow",f"{side}_wrist",0.30)
        limb(f"{side}_thigh",f"{side}_hip",f"{side}_knee",0.40)
        limb(f"{side}_calf",f"{side}_knee",f"{side}_ankle",0.32)
        if include_hands:
            wrist=point(pose,f"{side}_wrist",threshold); cloud=_cloud(pose.get(f"{side}_hand_points",[]),threshold)
            if wrist is not None and len(cloud)>=5:
                pts=np.vstack([cloud,wrist]); mask=polygon_mask(shape,pts,max(3,int(np.linalg.norm(np.ptp(cloud,axis=0))*0.12)))&person_mask
                quality="ok" if len(cloud)>=12 else "uncertain"; parts.append(part_record(f"{side}_hand",mask,float(np.mean([p["confidence"] for p in pose[f"{side}_hand_points"] if p["confidence"]>=threshold])),"person_01",wrist,quality=quality))
            else: missing.append(f"{side}_hand")
        if include_feet:
            ankle=point(pose,f"{side}_ankle",threshold); names=[f"{side}_big_toe",f"{side}_small_toe",f"{side}_heel"]
            cloud=np.array([point(pose,n,threshold) for n in names if point(pose,n,threshold) is not None])
            if ankle is not None and len(cloud)>=2:
                pts=np.vstack([cloud,ankle]); mask=polygon_mask(shape,pts,max(4,int(np.linalg.norm(np.ptp(cloud,axis=0))*0.18)))&person_mask
                parts.append(part_record(f"{side}_foot",mask,_confidence(pose,names+[f"{side}_ankle"]),"person_01",ankle,quality="ok" if len(cloud)==3 else "partial"))
            else: missing.append(f"{side}_foot")
    return parts, missing


def _resolve_overlaps(parts):
    if not parts: return 0,0
    stack=np.stack([p["mask"] for p in parts]); before=int((stack.sum(axis=0)>1).sum())
    distances=[]
    for part in parts:
        seed=np.full(stack.shape[1:],255,dtype=np.uint8)
        if "joint_start" in part and "joint_end" in part:
            cv2.line(seed,tuple(np.rint(part["joint_start"]).astype(int)),tuple(np.rint(part["joint_end"]).astype(int)),0,1)
        else:
            cv2.circle(seed,tuple(np.rint(part["anchor"]).astype(int)),1,0,-1)
        distance=cv2.distanceTransform(seed,cv2.DIST_L2,3); distance[~part["mask"]]=np.inf; distances.append(distance)
    distance_stack=np.stack(distances)
    owner=np.argmin(distance_stack,axis=0).astype(np.int16)
    owner[~stack.any(axis=0)]=-1
    for idx,p in enumerate(parts): p["mask"]=owner==idx; p["area"]=int(p["mask"].sum()); p["bbox"]=_bbox(p["mask"])
    after=int((np.stack([p["mask"] for p in parts]).sum(axis=0)>1).sum())
    return before,after


def _bbox(mask):
    ys,xs=np.nonzero(mask)
    return [float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)] if len(xs) else [0,0,0,0]


def _body_preview(image,parts):
    base=np.asarray(image).copy(); overlay=base.copy()
    for i,p in enumerate(parts): overlay[p["mask"]]=PART_COLORS[i%len(PART_COLORS)]
    out=Image.fromarray(cv2.addWeighted(base,0.58,overlay,0.42,0)); draw=ImageDraw.Draw(out)
    for i,p in enumerate(parts):
        x,y=p["anchor"]; draw.text((x,y),p["label"].upper(),fill=PART_COLORS[i%len(PART_COLORS)])
    return pil_to_comfy_image(out)


def _exploded(image,parts):
    tile=280; cols=4; rows=max(1,math.ceil(len(parts)/cols)); canvas=Image.new("RGBA",(cols*tile,rows*tile),(25,25,25,255)); draw=ImageDraw.Draw(canvas)
    rgba=image.convert("RGBA")
    for i,p in enumerate(parts):
        x1,y1,x2,y2=map(int,p["bbox"]); 
        if x2<=x1 or y2<=y1: continue
        crop=rgba.crop((x1,y1,x2,y2)); alpha=Image.fromarray(p["mask"][y1:y2,x1:x2].astype(np.uint8)*255); crop.putalpha(alpha); crop.thumbnail((tile-20,tile-40))
        x=(i%cols)*tile+(tile-crop.width)//2; y=(i//cols)*tile+8; canvas.alpha_composite(crop,(x,y)); draw.text(((i%cols)*tile+8,(i//cols+1)*tile-25),p["label"],fill="white")
    return pil_to_comfy_image(canvas.convert("RGB"))


class KBLBodySplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image":("IMAGE",),"elements":("KBL_ELEMENTS",),"pose_data":("KBL_POSE",),"split_detail_level":(["standard","basic","fine"],{"default":"standard"}),"joint_confidence_threshold":("FLOAT",{"default":0.30,"min":0.05,"max":0.95,"step":0.01}),"sam_refine_parts":("BOOLEAN",{"default":True}),"resolve_overlap":("BOOLEAN",{"default":True}),"include_head":("BOOLEAN",{"default":True}),"include_hands":("BOOLEAN",{"default":True}),"include_feet":("BOOLEAN",{"default":True}),"include_face":("BOOLEAN",{"default":False}),"include_hair":("BOOLEAN",{"default":False})}}
    RETURN_TYPES=("KBL_BODY_PARTS","MASK","STRING","IMAGE","IMAGE","IMAGE")
    RETURN_NAMES=("body_parts","body_part_masks","diagnostics","body_preview","body_mask_sheet","exploded_preview")
    FUNCTION="split"; CATEGORY="Kitao Body Collage/人体"
    def split(self,image,elements,pose_data,split_detail_level,joint_confidence_threshold,sam_refine_parts,resolve_overlap,include_head,include_hands,include_feet,include_face,include_hair):
        pil=comfy_image_to_pil(image); selected=pose_data.get("selected_person"); pose=pose_data.get("selected_pose")
        if not selected or not pose:
            diag={"selected_person_id":None,"valid_body_parts":[],"missing_parts":STANDARD_PARTS,"body_part_count":0}
            return [],masks_to_batch([],pil.height,pil.width),json.dumps(diag,ensure_ascii=False),image,render_mask_contact_sheet([]),render_mask_contact_sheet([])
        person_mask=np.asarray(selected["mask"],dtype=bool); parts,missing=_candidate_parts(pose,person_mask,joint_confidence_threshold,include_head,include_hands,include_feet)
        if split_detail_level=="basic":
            groups={"left_arm":["left_upper_arm","left_forearm","left_hand"],"right_arm":["right_upper_arm","right_forearm","right_hand"],"left_leg":["left_thigh","left_calf","left_foot"],"right_leg":["right_thigh","right_calf","right_foot"]}
            base=[p for p in parts if p["label"] in {"head","torso"}]
            for label,names in groups.items():
                source=[p for p in parts if p["label"] in names]
                if source:
                    mask=np.logical_or.reduce([p["mask"] for p in source]); base.append(part_record(label,mask,min(p["confidence"] for p in source),selected["id"],source[0]["anchor"],quality="ok"))
            parts=base
        for p in parts: p["source_person_id"]=selected["id"]
        if sam_refine_parts and parts:
            prompts=[{"bbox":p["bbox"]} for p in parts]; sam=Sam2Backend(Path(get_path_config()["model_root"])/"Kitao_Body_Collage_Lab"/"sam2"); refined=sam.refine_part_prompts(pil,prompts)
            for p,r in zip(parts,refined):
                constrained=r["mask"]&p["mask"]&person_mask
                if constrained.sum()>=max(8,p["mask"].sum()*0.20): p["mask"]=constrained; p["sam_score"]=r["sam_score"]; p["source"]="dwpose+sam2"; p["area"]=int(constrained.sum()); p["bbox"]=_bbox(constrained)
        before,after=_resolve_overlaps(parts) if resolve_overlap else (int((np.stack([p["mask"] for p in parts]).sum(axis=0)>1).sum()) if parts else 0,0)
        parts=[p for p in parts if p["area"]>0]
        union=np.logical_or.reduce([p["mask"] for p in parts]) if parts else np.zeros_like(person_mask)
        uncertain=[p["label"] for p in parts if p["quality_flag"]!="ok"]
        pose_collections = [
            pose.get("keypoints", {}).values(),
            pose.get("face_points", []),
            pose.get("left_hand_points", []),
            pose.get("right_hand_points", []),
        ]
        detected_pose_points = sum(
            1
            for collection in pose_collections
            for value in collection
            if float(value.get("confidence", 0.0)) >= joint_confidence_threshold
        )
        diag={"selected_person_id":selected["id"],"person_mask_area":int(person_mask.sum()),"detected_pose_points":detected_pose_points,"valid_body_parts":[p["label"] for p in parts],"missing_parts":missing,"uncertain_parts":uncertain,"body_part_count":len(parts),"overlap_pixels_before":before,"overlap_pixels_after":after,"union_area":int(union.sum()),"union_vs_person_ratio":float(union.sum()/max(1,person_mask.sum())),"split_detail_level":split_detail_level,"fine_experimental":split_detail_level=="fine","face_requested_but_not_emitted":bool(include_face),"hair_requested_but_not_emitted":bool(include_hair)}
        return parts,masks_to_batch(parts,pil.height,pil.width),json.dumps(diag,ensure_ascii=False),_body_preview(pil,parts),render_mask_contact_sheet(parts),_exploded(pil,parts)

NODE_CLASS_MAPPINGS={"KBL_Body_Splitter":KBLBodySplitter}
NODE_DISPLAY_NAME_MAPPINGS={"KBL_Body_Splitter":"KBL 人体部位拆分"}
