"""Atomic, deterministic KBL collage asset-package export."""

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..version import KBL_VERSION
from .utils.alpha_utils import mask_bbox
from .utils.image_io import comfy_image_to_pil, pil_to_comfy_image
from .utils.manifest_utils import KBL_MANIFEST_VERSION, validate_manifest
from .utils.path_config import get_path_config

CATEGORY_FOLDERS = {
    "person": "person", "clothing": "clothing", "shoe": "clothing",
    "chair": "furniture", "table": "furniture",
    "flower": "plants", "plant": "plants",
    "bag": "props", "glass": "props", "mirror": "props", "rope": "props", "jewelry": "props",
}
CATEGORY_PREFIXES = {"person": "person", "clothing": "clothing", "furniture": "furniture", "plants": "plant", "props": "prop", "other": "element", "unlabeled": "region"}
BODY_ORDER = [
    "head", "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm", "right_forearm", "right_hand",
    "torso", "left_thigh", "left_calf", "left_foot", "right_thigh", "right_calf", "right_foot",
]


def validate_project_name(name):
    value = str(name).strip()
    if not value or len(value) > 80 or value in {".", ".."}:
        raise ValueError("KBL project_name 不能为空、不能超过 80 字符")
    if any(char in '<>:"/\\|?*' or ord(char) < 32 for char in value):
        raise ValueError("KBL project_name 含 Windows 非法字符")
    if value[-1] in {".", " "}:
        raise ValueError("KBL project_name 不能以点或空格结尾")
    return value


def category_folder(item):
    label = str(item.get("label", "")).lower()
    category = str(item.get("category", "")).lower()
    if label in CATEGORY_FOLDERS:
        return CATEGORY_FOLDERS[label]
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", label)))
    for token in ("person", "clothing", "shoe", "chair", "table", "flower", "plant", "bag", "glass", "mirror", "rope", "jewelry"):
        if token in tokens:
            return CATEGORY_FOLDERS[token]
    if not item.get("semantic", True) or label.startswith("region_"):
        return "unlabeled"
    return category if category in CATEGORY_PREFIXES else "other"


def padded_crop_bbox(content_bbox, width, height, padding):
    x1, y1, x2, y2 = [int(round(value)) for value in content_bbox]
    pad = max(0, int(padding))
    return [max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)]


def to_local_anchor(anchor, crop_origin):
    return [float(anchor[0]) - float(crop_origin[0]), float(anchor[1]) - float(crop_origin[1])]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose_project_path(root, project_name, policy):
    desired = root / project_name
    if not desired.exists():
        return desired
    if policy == "replace":
        return desired
    if policy == "skip":
        raise FileExistsError(f"KBL 项目已存在且 overwrite_policy=skip：{desired}")
    version = 2
    while True:
        candidate = root / f"{project_name}_v{version:03d}"
        if not candidate.exists():
            return candidate
        version += 1


def _save_mask(mask, path, alpha=False):
    array = np.asarray(mask)
    if alpha:
        pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        pixels = (array > 0).astype(np.uint8) * 255
    Image.fromarray(pixels, mode="L").save(path)


def _overview(source, assets, refined_by_key):
    image = source.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [(239,71,111),(255,209,102),(6,214,160),(17,138,178),(144,190,109),(244,162,97)]
    for index, (kind, item) in enumerate(assets):
        record = refined_by_key[(kind, item["id"])]
        mask = np.asarray(record["binary_mask"], dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[index % len(colors)]
        for contour in contours:
            points = [tuple(map(int, point[0])) for point in contour[::max(1, len(contour)//300)]]
            if len(points) > 1:
                draw.line(points + [points[0]], fill=color, width=max(2, source.width // 700))
        bbox = mask_bbox(mask)
        draw.text((bbox[0] + 2, max(0, bbox[1] - 14)), f"{index+1} {item['label']}", fill=color, font=font)
    return image


def _checker(size, block=16):
    width, height = size
    y, x = np.indices((height, width))
    grid = ((x // block + y // block) % 2)[..., None]
    pixels = np.where(grid, np.array([205,205,205]), np.array([238,238,238])).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _asset_sheet(root, entries, title, columns=4, tile=300):
    count = max(1, len(entries)); rows = (count + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile + 48), "#202020")
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    draw.text((16, 16), title, fill="white", font=font)
    for index, item in enumerate(entries):
        with Image.open(root / item["file"]) as opened:
            asset = opened.convert("RGBA")
        asset.thumbnail((tile - 28, tile - 66), Image.Resampling.LANCZOS)
        x0=(index%columns)*tile; y0=48+(index//columns)*tile
        background=_checker((tile-16,tile-48)); sheet.paste(background,(x0+8,y0+4))
        x=x0+(tile-asset.width)//2; y=y0+8+(tile-58-asset.height)//2
        sheet.paste(asset,(x,y),asset)
        with Image.open(root / item["file"]) as dimensions:
            width,height=dimensions.size
        source=str(item.get("discovery_source",item.get("source","unknown"))).replace("florence_", "Florence ")
        quality=str(item.get("quality_flag","ok")).upper()
        draw.text((x0+10,y0+tile-39),f"{item['label'].upper()}  {width}x{height}",fill="white",font=font)
        draw.text((x0+10,y0+tile-24),f"{source} + SAM2 | {quality}",fill="#b9c3c9",font=font)
    return sheet


def _mask_sheet(records, columns=4, tile=240):
    count=max(1,len(records)); rows=(count+columns-1)//columns
    sheet=Image.new("RGB",(columns*tile,rows*tile),"#181818"); draw=ImageDraw.Draw(sheet); font=ImageFont.load_default()
    for index,record in enumerate(records):
        mask=Image.fromarray((np.asarray(record["alpha_mask"])*255).astype(np.uint8),mode="L")
        mask.thumbnail((tile-16,tile-40),Image.Resampling.NEAREST)
        x=(index%columns)*tile+(tile-mask.width)//2; y=(index//columns)*tile+6
        sheet.paste(Image.new("RGB",mask.size,"#47d7ac"),(x,y),mask)
        draw.text(((index%columns)*tile+8,(index//columns+1)*tile-28),record["label"],fill="white",font=font)
    return sheet


def _element_filename(item, folder):
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(item.get("id", "auto_01"))).strip("_") or "auto_01"
    return f"{safe_id[:96]}.png"


def _anchor(item):
    if item.get("original_anchor") is not None:
        return [float(value) for value in item["original_anchor"]]
    if item.get("anchor") is not None:
        return [float(value) for value in item["anchor"]]
    x1,y1,x2,y2=item["bbox"]
    return [(float(x1)+float(x2))/2,(float(y1)+float(y2))/2]


def _write_asset(source, temp_root, kind, item, refined, crop_mode, padding, save_raw, save_refined):
    width,height=source.size
    raw=np.asarray(refined["raw_mask"],dtype=bool); binary=np.asarray(refined["binary_mask"],dtype=bool); alpha=np.asarray(refined["alpha_mask"],dtype=np.float32)
    for name,value in (("raw",raw),("binary",binary),("alpha",alpha)):
        if value.shape != (height,width):
            raise ValueError(f"{item['id']} {name} mask 尺寸 {value.shape} != {(height,width)}")
    if not np.any(alpha>0):
        raise ValueError(f"{item['id']} alpha 为空")
    content_bbox=mask_bbox(alpha>0)
    crop_bbox=[0,0,width,height] if crop_mode=="full_canvas" else padded_crop_bbox(content_bbox,width,height,padding)
    x1,y1,x2,y2=crop_bbox
    if kind=="body":
        relative=Path("body")/f"{item['id']}.png"
    else:
        folder=category_folder(item); relative=Path("elements")/folder/_element_filename(item,folder)
    target=temp_root/relative; target.parent.mkdir(parents=True,exist_ok=True)
    source_crop=source.crop((x1,y1,x2,y2)).convert("RGBA")
    source_crop.putalpha(Image.fromarray(np.rint(np.clip(alpha[y1:y2,x1:x2],0,1)*255).astype(np.uint8),mode="L"))
    source_crop.save(target,format="PNG")
    full_canvas_file=None
    if crop_mode=="both":
        full_relative=relative.with_name(relative.stem+"_full_canvas.png")
        full=source.copy().convert("RGBA"); full.putalpha(Image.fromarray(np.rint(np.clip(alpha,0,1)*255).astype(np.uint8),mode="L")); full.save(temp_root/full_relative,format="PNG")
        full_canvas_file=full_relative.as_posix()
    mask_id=re.sub(r"[^A-Za-z0-9_-]+","_",str(item["id"]))
    raw_path=Path("masks/raw")/f"{mask_id}.png" if save_raw else None
    refined_path=Path("masks/refined")/f"{mask_id}.png" if save_refined else None
    alpha_path=Path("masks/alpha")/f"{mask_id}.png" if save_refined else None
    if raw_path: _save_mask(raw,temp_root/raw_path)
    if refined_path: _save_mask(binary,temp_root/refined_path)
    if alpha_path: _save_mask(alpha,temp_root/alpha_path,alpha=True)
    original_anchor=_anchor(item); local_anchor=to_local_anchor(original_anchor,[x1,y1])
    record={
        "id":item["id"],"label":item["label"],"category":item.get("category",kind),"file":relative.as_posix(),
        "full_canvas_file":full_canvas_file,"raw_mask_file":raw_path.as_posix() if raw_path else None,
        "refined_mask_file":refined_path.as_posix() if refined_path else None,"alpha_mask_file":alpha_path.as_posix() if alpha_path else None,
        "original_bbox":[float(v) for v in item["bbox"]],"content_bbox":content_bbox,"crop_bbox":crop_bbox,"crop_origin":[x1,y1],
        "area":int(raw.sum()),"raw_area":int(raw.sum()),"refined_area":int(binary.sum()),
        "area_change_percent":float((binary.sum()-raw.sum())/max(1,raw.sum())*100.0),
        "confidence":float(item.get("confidence",0.0)),
        "pose_confidence":float(item.get("pose_confidence",item.get("confidence",0.0))),"sam_score":float(item.get("sam_score",0.0)),
        "quality_flag":item.get("quality_flag","ok"),"source_person_id":item.get("source_person_id"),
        "original_anchor":original_anchor,"local_anchor":local_anchor,"orientation_deg":float(item.get("orientation_deg",0.0)),
        "joint_start":item.get("joint_start"),"joint_end":item.get("joint_end"),"source":item.get("source","unknown"),
    }
    if kind=="element":
        record.update({key:item.get(key) for key in (
            "raw_label", "canonical_label", "discovery_source", "semantic",
            "fragment_score", "whole_object_score",
        ) if key in item})
        record["mask"]=record["raw_mask_file"]; record["bbox"]=record["original_bbox"]
        record["original_position"]=[(record["original_bbox"][0]+record["original_bbox"][2])/2,(record["original_bbox"][1]+record["original_bbox"][3])/2]
    return record


class KBLCutoutExporter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "image_meta": ("STRING",), "elements": ("KBL_ELEMENTS",),
            "body_parts": ("KBL_BODY_PARTS",), "refined_masks": ("KBL_REFINED_MASKS",),
            "project_name": ("STRING", {"default":"KBL_PROJECT"}),
            "export_root": ("STRING", {"default":"N:\\ComfyUI\\output\\Kitao_Body_Collage_Lab"}),
            "export_scope": (["all","body_only","elements_only"], {"default":"all"}),
            "copy_source": ("BOOLEAN", {"default":True}), "save_raw_masks": ("BOOLEAN", {"default":True}),
            "save_refined_masks": ("BOOLEAN", {"default":True}), "save_preview": ("BOOLEAN", {"default":True}),
            "save_exploded_view": ("BOOLEAN", {"default":True}), "save_manifest": ("BOOLEAN", {"default":True}),
            "crop_mode": (["cropped","full_canvas","both"], {"default":"cropped"}),
            "padding": ("INT", {"default":24,"min":0,"max":512}),
            "overwrite_policy": (["version","replace","skip"], {"default":"version"}),
            "min_export_area": ("INT", {"default":64,"min":0,"max":1000000}),
        }}
    RETURN_TYPES=("STRING","STRING","INT","INT","INT","IMAGE","STRING","IMAGE")
    RETURN_NAMES=("project_directory","manifest_path","exported_count","body_exported_count","element_exported_count","preview","diagnostics","exploded_view")
    FUNCTION="export"; CATEGORY="Kitao Body Collage/导出"; OUTPUT_NODE=True

    def export(self,image,image_meta,elements,body_parts,refined_masks,project_name,export_root,export_scope,copy_source,save_raw_masks,save_refined_masks,save_preview,save_exploded_view,save_manifest,crop_mode,padding,overwrite_policy,min_export_area):
        project_name=validate_project_name(project_name); source=comfy_image_to_pil(image).convert("RGB"); width,height=source.size
        root=Path(export_root.strip() or get_path_config()["export_root"]).resolve(); root.mkdir(parents=True,exist_ok=True)
        if not os.access(root,os.W_OK): raise PermissionError(f"KBL 导出目录不可写：{root}")
        if shutil.disk_usage(root).free < max(64*1024*1024,width*height*4): raise OSError("KBL 导出磁盘空间不足")
        final=_choose_project_path(root,project_name,overwrite_policy); temp=root/f".{final.name}.kbl_tmp_{uuid.uuid4().hex}"
        backup=None
        try:
            for directory in ("source","body","elements/person","elements/clothing","elements/props","elements/furniture","elements/plants","elements/other","elements/unlabeled","masks/raw","masks/refined","masks/alpha","preview"):
                (temp/directory).mkdir(parents=True,exist_ok=True)
            try: meta=json.loads(image_meta) if image_meta else {}
            except json.JSONDecodeError as exc: raise ValueError(f"image_meta 不是合法 JSON：{exc}") from exc
            source_path=Path(meta.get("source_path","")) if meta.get("source_path") else None
            copied=None
            if copy_source:
                if source_path and source_path.is_file():
                    copied=Path("source")/("original"+source_path.suffix.lower()); shutil.copy2(source_path,temp/copied); hash_path=source_path
                else:
                    copied=Path("source/original.png"); source.save(temp/copied,format="PNG"); hash_path=temp/copied
            else:
                if not source_path or not source_path.is_file(): raise FileNotFoundError("copy_source=false 时 source_path 必须存在，才能记录 SHA256")
                hash_path=source_path
            refined_by_key={(record["kind"],record["id"]):record for record in refined_masks.get("records",[])}
            assets=[]
            if export_scope in {"all","elements_only"}:
                assets += [("element",item) for item in elements if int(item.get("area",0))>=int(min_export_area)]
            if export_scope in {"all","body_only"}:
                assets += [("body",item) for item in body_parts if int(item.get("area",0))>0]
            manifest_elements=[]; manifest_body=[]
            for kind,item in assets:
                key=(kind,item["id"])
                if key not in refined_by_key: raise ValueError(f"refined_masks 缺少 {kind}:{item['id']}")
                entry=_write_asset(source,temp,kind,item,refined_by_key[key],crop_mode,padding,save_raw_masks,save_refined_masks)
                (manifest_body if kind=="body" else manifest_elements).append(entry)
            all_entries=manifest_body+manifest_elements
            overview=_overview(source,assets,refined_by_key)
            body_assets=[asset for asset in assets if asset[0]=="body"]
            body_overview=_overview(source,body_assets,refined_by_key)
            contact=_asset_sheet(temp,all_entries,"KBL COLLAGE ASSET CONTACT SHEET")
            ordered_body=sorted(manifest_body,key=lambda item: BODY_ORDER.index(item["label"]) if item["label"] in BODY_ORDER else 999)
            exploded=_asset_sheet(temp,ordered_body+manifest_elements,"KBL EXPLODED ASSET VIEW")
            mask_records=[refined_by_key[(kind,item["id"])] for kind,item in assets]
            masks=_mask_sheet(mask_records)
            if save_preview:
                overview.save(temp/"preview/overview.png"); body_overview.save(temp/"preview/body_parts_preview.png")
                masks.save(temp/"preview/mask_sheet.png"); contact.save(temp/"preview/contact_sheet.png")
            if save_exploded_view: exploded.save(temp/"preview/exploded_view.png")
            person_masks=[np.asarray(item["mask"],dtype=bool) for item in elements if item.get("label")=="person"]
            person_mask=max(person_masks,key=lambda value:int(value.sum())) if person_masks else None
            body_subset_person=all(not np.any(np.asarray(item["mask"],dtype=bool)&~person_mask) for item in body_parts) if person_mask is not None else not body_parts
            overlap_after=int((np.stack([np.asarray(item["mask"],dtype=bool) for item in body_parts]).sum(axis=0)>1).sum()) if body_parts else 0
            pipeline_diagnostics={
                "pipeline_version":KBL_VERSION,
                "source":meta.get("filename",source_path.name if source_path else "original.png"),
                "resolution":[width,height],
                "timings":{"grounding_dino":0.0,"sam2":0.0,"dwpose":0.0,"body_split":0.0,"refine":0.0,"export":0.0,"total":0.0},
                "counts":{"elements":len(manifest_elements),"body_parts":len(manifest_body),"missing_parts":len(refined_masks.get("missing_body_parts",[])),"uncertain_parts":sum(1 for item in manifest_body if item.get("quality_flag")!="ok")},
                "validation":{"body_subset_person":body_subset_person,"overlap_after":overlap_after,"manifest_valid":False},
                "refine":[{"id":item["id"],"raw_area":item["raw_area"],"refined_area":item["refined_area"],"area_change_percent":item["area_change_percent"],"warning":"REFINE_AREA_WARNING" if abs(item["area_change_percent"])>15.0 else None} for item in all_entries],
            }
            manifest={
                "manifest_version":KBL_MANIFEST_VERSION,"pipeline_version":KBL_VERSION,"project_name":final.name,
                "created_at":datetime.now(timezone.utc).isoformat(),
                "source":{"filename":meta.get("filename",source_path.name if source_path else "original.png"),"source_path":str(source_path) if source_path else None,"copied_source_path":copied.as_posix() if copied else None,"sha256":_sha256(hash_path)},
                "image":{"width":width,"height":height,"format":meta.get("format",source_path.suffix.lstrip(".").upper() if source_path else "PNG"),"orientation":meta.get("orientation",1),"color_mode":"RGB"},
                "elements":manifest_elements,"body_parts":manifest_body,
                "diagnostics":{"export_scope":export_scope,"crop_mode":crop_mode,"padding":int(padding),"body_exported_count":len(manifest_body),"element_exported_count":len(manifest_elements),"missing_body_parts":refined_masks.get("missing_body_parts",[]),"birefnet":refined_masks.get("birefnet_status","NOT INSTALLED"),"models_rerun":False,"pipeline_diagnostics_file":"pipeline_diagnostics.json"},
                "export_complete":True,
            }
            validator={"status":"NOT_REQUESTED","errors":[],"checked_assets":0}
            manifest_file=temp/"manifest.json"
            if save_manifest:
                manifest_file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
                validator=validate_manifest(manifest_file)
                if validator["status"]!="PASS": raise RuntimeError("KBL manifest validator FAILED: "+"; ".join(validator["errors"]))
            pipeline_diagnostics["validation"]["manifest_valid"]=validator["status"]=="PASS"
            (temp/"pipeline_diagnostics.json").write_text(json.dumps(pipeline_diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
            if final.exists():
                backup=root/f".{final.name}.kbl_backup_{uuid.uuid4().hex}"; os.replace(final,backup)
            os.replace(temp,final)
            if backup and backup.exists(): shutil.rmtree(backup)
            manifest_path=str(final/"manifest.json") if save_manifest else ""
            diagnostics={"status":"PASS","validator":validator,"project":str(final),"exported_count":len(all_entries),"models_rerun":False}
            print(f"[KBL Export]\nProject:\n{final}\nBody parts:\n{len(manifest_body)}\nElements:\n{len(manifest_elements)}\nOutput:\n{final}\nManifest:\n{manifest_path}")
            return str(final),manifest_path,len(all_entries),len(manifest_body),len(manifest_elements),pil_to_comfy_image(contact),json.dumps(diagnostics,ensure_ascii=False),pil_to_comfy_image(exploded)
        except Exception:
            if temp.exists(): shutil.rmtree(temp)
            if backup and backup.exists() and not final.exists(): os.replace(backup,final)
            raise


NODE_CLASS_MAPPINGS={"KBL_Cutout_Exporter":KBLCutoutExporter}
NODE_DISPLAY_NAME_MAPPINGS={"KBL_Cutout_Exporter":"KBL 拼贴素材导出"}
