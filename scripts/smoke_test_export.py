"""Full real Stage D smoke test ending in a validated portable asset package."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from Kitao_Body_Collage_Lab.nodes.body_split_nodes import KBLBodySplitter
from Kitao_Body_Collage_Lab.nodes.detection_nodes import KBLElementDetector
from Kitao_Body_Collage_Lab.nodes.export_nodes import KBLCutoutExporter
from Kitao_Body_Collage_Lab.nodes.input_nodes import KBLLoadImage
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.refine_nodes import KBLMaskRefiner
from Kitao_Body_Collage_Lab.nodes.utils.image_io import comfy_image_to_pil
from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest

from model_integrity import require_models

MODEL_ROOT=Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab")
OUTPUT_ROOT=Path("N:/ComfyUI/output/Kitao_Body_Collage_Lab/validation_stage_d")
KEY_LABELS=("head","torso","left_hand","right_hand","left_foot","right_foot")


def inspect_png(project, item, source):
    path=project/item["file"]
    with Image.open(path) as opened:
        if opened.mode!="RGBA": raise RuntimeError(f"{item['label']} 不是 RGBA")
        rgba=np.asarray(opened.convert("RGBA")); size=list(opened.size)
    alpha=rgba[...,3]; ys,xs=np.nonzero(alpha)
    if not len(xs): raise RuntimeError(f"{item['label']} alpha 为空")
    x0,y0=item["crop_origin"]; source_pixels=np.asarray(source)
    sample_index=len(xs)//2; local_x,local_y=int(xs[sample_index]),int(ys[sample_index])
    if not np.array_equal(rgba[local_y,local_x,:3],source_pixels[y0+local_y,x0+local_x]):
        raise RuntimeError(f"{item['label']} RGB 与原图不一致")
    anchor=item["local_anchor"]
    if not (0<=anchor[0]<=size[0] and 0<=anchor[1]<=size[1]): raise RuntimeError(f"{item['label']} local_anchor 越界")
    return {"file":item["file"],"size":size,"alpha_bbox":[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],"alpha_nonzero":int(len(xs)),"quality":item["quality_flag"],"anchor_ok":True,"rgb_matches_source":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",required=True)
    args=parser.parse_args(); started=time.perf_counter()
    require_models(MODEL_ROOT); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    image,image_meta,*_=KBLLoadImage().load(args.image); source=comfy_image_to_pil(image)
    elements,*_=KBLElementDetector().detect(image,"guided","portrait_basic","person.",.25,16,.85,64,16)
    if not any(item["label"]=="person" for item in elements): raise RuntimeError("没有 person element")
    pose_data,_,_=KBLPoseEstimator().estimate(image,elements,"largest",0,.30)
    body_parts,_,body_diag,*_=KBLBodySplitter().split(image,elements,pose_data,"standard",.30,True,True,True,True,True,False,False)
    if len(body_parts)!=14: raise RuntimeError(f"body part 数量不是 14：{len(body_parts)}")
    refined,_,_,refine_diag=KBLMaskRefiner().refine(image,elements,body_parts,"safe",1,0,.75,16,True,True)
    soft,_,_,soft_diag=KBLMaskRefiner().refine(image,elements,body_parts,"soft",1,0,2.0,16,True,True)
    result=KBLCutoutExporter().export(image,image_meta,elements,body_parts,refined,"KBL_STAGE_D_TEST",str(OUTPUT_ROOT),"all",True,True,True,True,True,True,"cropped",24,"replace",64)
    project=Path(result[0]); manifest_path=Path(result[1]); manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    validator=validate_manifest(manifest_path)
    if validator["status"]!="PASS": raise RuntimeError(json.dumps(validator,ensure_ascii=False))
    by_label={item["label"]:item for item in manifest["body_parts"]}
    key_png={label:inspect_png(project,by_label[label],source) for label in KEY_LABELS}
    soft_records=soft["records"]
    partial_alpha=sum(int(np.count_nonzero((item["alpha_mask"]>0)&(item["alpha_mask"]<1))) for item in soft_records)
    if partial_alpha<=0: raise RuntimeError("soft alpha 没有 0~1 边缘像素")
    smoke={
        "status":"PASS","image":json.loads(image_meta),"project_directory":str(project),"manifest_path":str(manifest_path),
        "body_exported_count":result[3],"element_exported_count":result[4],"exported_count":result[2],
        "body_diagnostics":json.loads(body_diag),"safe_refine":json.loads(refine_diag),
        "soft_refine":{"diagnostics":json.loads(soft_diag),"partial_alpha_pixels":partial_alpha},
        "key_png_validation":key_png,"manifest_validator":validator,
        "previews":{name:str(project/"preview"/name) for name in ("overview.png","body_parts_preview.png","mask_sheet.png","exploded_view.png","contact_sheet.png")},
        "elapsed_seconds":round(time.perf_counter()-started,4),"mock_or_placeholder":False,"models_rerun_by_exporter":False,
    }
    result_path=OUTPUT_ROOT/"stage_d_smoke_result.json"; result_path.write_text(json.dumps(smoke,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(smoke,ensure_ascii=False,indent=2)); print(f"Saved: {result_path}")


if __name__=="__main__": main()
