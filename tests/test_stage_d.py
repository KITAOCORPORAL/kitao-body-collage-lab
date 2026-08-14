import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from Kitao_Body_Collage_Lab.version import KBL_VERSION

from Kitao_Body_Collage_Lab.nodes.export_nodes import (
    KBLCutoutExporter, category_folder, padded_crop_bbox, to_local_anchor,
)
from Kitao_Body_Collage_Lab.nodes.refine_nodes import KBLMaskRefiner
from Kitao_Body_Collage_Lab.nodes.utils.alpha_utils import refine_binary_mask, soft_alpha_from_binary
from Kitao_Body_Collage_Lab.nodes.utils.image_io import pil_to_comfy_image
from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest


def fixture(root):
    source = Image.new("RGB", (80, 100))
    pixels = np.asarray(source).copy()
    pixels[..., 0] = np.arange(80, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.arange(100, dtype=np.uint8)[:, None]
    pixels[..., 2] = 127
    source = Image.fromarray(pixels, "RGB")
    source_path = root / "fixture.jpg"
    source.save(source_path, quality=95)
    person = np.zeros((100, 80), bool); person[5:96, 10:71] = True
    flower = np.zeros_like(person); flower[15:30, 3:18] = True
    head = np.zeros_like(person); head[8:30, 27:53] = True
    hand = np.zeros_like(person); hand[45:55, 58:68] = True
    elements = [
        {"id":"person_01","label":"person","category":"person","mask":person,"bbox":[10,5,71,96],"area":int(person.sum()),"confidence":.96,"sam_score":.91,"source":"guided"},
        {"id":"flower_01","label":"flower","category":"prop","mask":flower,"bbox":[3,15,18,30],"area":int(flower.sum()),"confidence":.80,"sam_score":.88,"source":"guided"},
    ]
    body = [
        {"id":"body_head","label":"head","category":"body","mask":head,"bbox":[27,8,53,30],"area":int(head.sum()),"confidence":.9,"pose_confidence":.9,"sam_score":.8,"quality_flag":"ok","source_person_id":"person_01","anchor":[40,18],"original_anchor":[40,18],"orientation_deg":0.0,"source":"dwpose+sam2"},
        {"id":"body_left_hand","label":"left_hand","category":"body","mask":hand,"bbox":[58,45,68,55],"area":int(hand.sum()),"confidence":.7,"pose_confidence":.7,"sam_score":.75,"quality_flag":"uncertain","source_person_id":"person_01","anchor":[60,48],"original_anchor":[60,48],"orientation_deg":12.5,"source":"dwpose+sam2"},
    ]
    meta=json.dumps({"filename":source_path.name,"source_path":str(source_path),"width":80,"height":100,"format":"JPEG","orientation":1})
    return source, meta, elements, body


def refine(source, elements, body, mode="safe"):
    return KBLMaskRefiner().refine(pil_to_comfy_image(source),elements,body,mode,1,0,2.0,4,True,True)[0]


class StageDTests(unittest.TestCase):
    def test_raw_mask_stays_unchanged_and_safe_refines(self):
        raw=np.zeros((30,30),bool); raw[5:25,5:25]=True; raw[14,14]=False; raw[1,1]=True
        snapshot=raw.copy(); result=refine_binary_mask(raw,1,0,4,True,True)
        self.assertTrue(np.array_equal(raw,snapshot)); self.assertTrue(result[14,14]); self.assertFalse(result[1,1])
        self.assertLess(abs(int(result.sum())-int(raw.sum()))/raw.sum(),.35)

    def test_soft_alpha_has_opaque_interior_gradient_and_zero_exterior(self):
        mask=np.zeros((40,40),bool); mask[10:30,10:30]=True
        alpha=soft_alpha_from_binary(mask,2.0)
        self.assertEqual(alpha[20,20],1.0); self.assertGreater(alpha[9,20],0.0); self.assertLess(alpha[9,20],1.0); self.assertEqual(alpha[0,0],0.0)

    def test_refiner_preserves_raw_and_body_subset(self):
        with tempfile.TemporaryDirectory() as td:
            source,_,elements,body=fixture(Path(td)); original=body[0]["mask"].copy()
            result=refine(source,elements,body,"soft"); record=next(x for x in result["records"] if x["id"]=="body_head")
            self.assertTrue(np.array_equal(body[0]["mask"],original)); self.assertFalse(np.any(record["binary_mask"] & ~elements[0]["mask"])); self.assertFalse(np.any((record["alpha_mask"]>0) & ~elements[0]["mask"]))

    def test_crop_padding_and_local_anchor(self):
        self.assertEqual(padded_crop_bbox([2,3,10,12],20,20,5),[0,0,15,17])
        self.assertEqual(to_local_anchor([12,15],[5,7]),[7.0,8.0])

    def test_category_mapping(self):
        self.assertEqual(category_folder({"label":"chair"}),"furniture"); self.assertEqual(category_folder({"label":"flower"}),"plants"); self.assertEqual(category_folder({"label":"foot_shoe"}),"clothing"); self.assertEqual(category_folder({"label":"unknown"}),"other")

    def _export(self, root, policy="version", body_override=None, project="TEST_PROJECT"):
        source,meta,elements,body=fixture(root); body=body_override if body_override is not None else body
        refined=refine(source,elements,body)
        return KBLCutoutExporter().export(pil_to_comfy_image(source),meta,elements,body,refined,project,str(root/"out"),"all",True,True,True,True,True,True,"cropped",6,policy,1)

    def test_export_rgba_straight_alpha_anchor_manifest_and_quality(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); result=self._export(root); project=Path(result[0]); manifest=json.loads(Path(result[1]).read_text(encoding="utf-8"))
            self.assertEqual(result[3],2); self.assertEqual(result[4],2); self.assertFalse(manifest["diagnostics"]["models_rerun"])
            pipeline=json.loads((project/"pipeline_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline["pipeline_version"], KBL_VERSION); self.assertTrue(pipeline["validation"]["manifest_valid"])
            self.assertIn("area_change_percent",pipeline["refine"][0])
            hand=next(x for x in manifest["body_parts"] if x["label"]=="left_hand")
            self.assertEqual(hand["quality_flag"],"uncertain"); self.assertEqual(hand["local_anchor"],to_local_anchor(hand["original_anchor"],hand["crop_origin"]))
            with Image.open(project/hand["file"]) as png:
                self.assertEqual(png.mode,"RGBA"); alpha=np.asarray(png.getchannel("A")); self.assertGreater(alpha.max(),0); self.assertGreater(png.width,10); self.assertGreater(png.height,10)
                rgba=np.asarray(png); self.assertTrue(np.any((alpha==0))); self.assertTrue(np.any(rgba[alpha==0,:3] != 0))
            self.assertEqual(validate_manifest(result[1])["status"],"PASS")

    def test_manifest_paths_are_relative_and_masks_exist(self):
        with tempfile.TemporaryDirectory() as td:
            result=self._export(Path(td)); manifest=json.loads(Path(result[1]).read_text(encoding="utf-8"))
            for item in manifest["body_parts"]+manifest["elements"]:
                self.assertFalse(Path(item["file"]).is_absolute()); self.assertFalse(Path(item["raw_mask_file"]).is_absolute())

    def test_missing_part_does_not_create_fake_png(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source,meta,elements,body=fixture(root); body=body[:1]; refined=refine(source,elements,body)
            result=KBLCutoutExporter().export(pil_to_comfy_image(source),meta,elements,body,refined,"MISSING_TEST",str(root/"out"),"body_only",True,True,True,False,False,True,"cropped",6,"version",1)
            manifest=json.loads(Path(result[1]).read_text(encoding="utf-8")); self.assertEqual(len(manifest["body_parts"]),1); self.assertIn("left_hand",manifest["diagnostics"]["missing_body_parts"])

    def test_version_policy_never_silently_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); first=self._export(root); second=self._export(root)
            self.assertNotEqual(first[0],second[0]); self.assertTrue(second[0].endswith("_v002"))

    def test_atomic_failure_leaves_no_project_or_tmp_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source,meta,elements,body=fixture(root); refined=refine(source,elements,body)
            refined["records"][0]["alpha_mask"]=np.zeros((2,2),np.float32)
            with self.assertRaisesRegex(ValueError,"尺寸"):
                KBLCutoutExporter().export(pil_to_comfy_image(source),meta,elements,body,refined,"BROKEN",str(root/"out"),"all",True,True,True,False,False,True,"cropped",6,"version",1)
            out=root/"out"; self.assertFalse((out/"BROKEN").exists()); self.assertEqual(list(out.glob("*.kbl_tmp*")),[])

    def test_refiner_birefnet_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            source,_,elements,body=fixture(Path(td))
            with self.assertRaisesRegex(RuntimeError,"NOT INSTALLED"):
                KBLMaskRefiner().refine(pil_to_comfy_image(source),elements,body,"birefnet",1,0,.75,4,True,True)


if __name__ == "__main__":
    unittest.main()
