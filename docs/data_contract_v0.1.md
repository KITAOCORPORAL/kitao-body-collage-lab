# KBL Data Contract v0.1

Kitao Body Collage Lab `0.1.0` freezes three public contracts: `KBL_ELEMENTS v0.1`, `KBL_BODY_PARTS v0.1`, and `KBL_MANIFEST v0.1`. Consumers may rely on the meanings below. A semantic change requires KBL `0.2` or a Manifest version increase; fields will not be silently redefined.

## Shared geometry rules

- Image origin is the top-left pixel.
- `x` increases to the right and `y` increases downward.
- Full-image masks use shape `[height, width]` and retain the oriented source image's dimensions.
- A bbox is `[x1, y1, x2, y2]` with a half-open lower-right edge: `x1 <= x < x2`, `y1 <= y < y2`.
- Full-image coordinates use the oriented source-image coordinate system after EXIF orientation is applied.
- Floating-point coordinates are allowed for pose-derived anchors and joints.

## KBL_ELEMENTS v0.1

Each in-memory element record contains at least:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable ID within one execution, such as `person_01`. |
| `label` | string | Detector label normalized to a short lower-case name. |
| `mask` | boolean array `[H,W]` | Independent original-size object mask. |
| `bbox` | number[4] | Full-image bbox enclosing `mask`. |
| `confidence` | number | Detection confidence in `[0,1]`. |
| `area` | integer | Count of non-zero mask pixels. |
| `source` | string | Detection path, such as `guided` or `auto`. |

Records may also carry `sam_score` and export metadata. Masks are not merged across objects.

## KBL_BODY_PARTS v0.1

Each in-memory body-part record contains at least:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Body asset ID, such as `body_left_forearm`. |
| `label` | string | Anatomical label; left/right refer to the person's anatomy. |
| `category` | string | `body`. |
| `mask` | boolean array `[H,W]` | Original-size visible-part mask; always a subset of the selected person mask. |
| `bbox` | number[4] | Full-image part bbox. |
| `area` | integer | Count of non-zero mask pixels. |
| `confidence` | number | Combined public confidence in `[0,1]`. |
| `pose_confidence` | number | DWPose confidence used for the part. |
| `sam_score` | number | SAM2 refinement score; `0` means the constrained geometric mask was retained. |
| `quality_flag` | string | `ok`, `uncertain`, `partial`, or `missing` when represented diagnostically. |
| `source_person_id` | string | ID of the parent person element. |
| `anchor` | number[2] | Full-image attachment/pivot point. |
| `original_anchor` | number[2] | Frozen full-image anchor retained through export. |
| `orientation_deg` | number | `atan2(joint_end.y-joint_start.y, joint_end.x-joint_start.x)` in degrees. Zero points right; because image y increases downward, positive values rotate visually clockwise. Non-limb parts use `0`. |
| `source` | string | Construction path, for example `dwpose+sam2`. |

Limb segments additionally contain `joint_start` and `joint_end`, both full-image `[x,y]` coordinates. Hand, foot, head, and torso records may use `null` joints after export.

## KBL_MANIFEST v0.1

Required top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `manifest_version` | string | Always `0.1` for this contract. |
| `pipeline_version` | string | Producer release, `0.1.0` for this baseline. |
| `project_name` | string | Exported directory name. |
| `created_at` | ISO-8601 string | UTC export time. |
| `source` | object | Source filename/path metadata and SHA-256; no image pixels. |
| `image` | object | Width, height, format, EXIF orientation, and color mode. |
| `elements` | array | Exported generic assets. |
| `body_parts` | array | Exported body assets. |
| `diagnostics` | object | Export settings and release diagnostics references. |
| `export_complete` | boolean | `true` only after the atomic export is complete. |

### Exported asset fields

`file`, `raw_mask_file`, `refined_mask_file`, and `alpha_mask_file` are POSIX-style paths relative to the project directory. Absolute paths, backslashes, and `..` traversal are not valid portable asset paths.

- `original_bbox`: bbox before refinement, in full-image coordinates.
- `content_bbox`: refined non-zero content bbox, in full-image coordinates.
- `crop_bbox`: exported crop in full-image coordinates.
- `crop_origin`: `[crop_bbox.x1, crop_bbox.y1]`.
- `original_anchor`: pivot in full-image coordinates.
- `local_anchor`: pivot in the cropped PNG, equal to `original_anchor - crop_origin`.
- `area` and `raw_area`: original mask pixel count; `area` is retained for v0.1 compatibility.
- `refined_area`: refined binary-mask pixel count.
- `area_change_percent`: `(refined_area - raw_area) / max(1, raw_area) * 100`.

RGBA files use straight alpha and preserve source RGB values at corresponding pixels. `pipeline_diagnostics.json` contains no image content; it records version, source filename, resolution, timings, counts, validation results, and per-mask refine-area diagnostics. `REFINE_AREA_WARNING` is emitted when the absolute area change exceeds 15%. Safe mode retains its automatic fallback when the initial change exceeds 20%.

## Compatibility policy

Readers must reject unsupported Manifest versions rather than guessing field semantics. Additive fields may be ignored by a v0.1 reader. Removing a required field, changing coordinate meaning, or changing a field type requires a new Manifest version or KBL v0.2 contract.
