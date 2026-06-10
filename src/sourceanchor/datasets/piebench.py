from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from sourceanchor.datasets.base import DatasetAdapter
from sourceanchor.outputs.writer import write_json
from sourceanchor.schemas import SampleMetadata, StandardSample


def decode_piebench_rle(rle: list[int], *, height: int = 512, width: int = 512) -> np.ndarray:
    """Decode PIE-Bench RLE [start0, len0, start1, len1, ...] into a binary mask."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    mask = np.zeros(height * width, dtype=np.uint8)
    for index in range(0, len(rle) - 1, 2):
        start = int(rle[index])
        length = int(rle[index + 1])
        if start < 0 or length < 0:
            raise ValueError("PIE-Bench RLE entries must be non-negative")
        if start >= mask.size or length == 0:
            continue
        mask[start : min(start + length, mask.size)] = 1
    mask = mask.reshape(height, width)
    # Match the original PIE-Bench/PnPI evaluation helper.
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask


class PIEBenchAdapter(DatasetAdapter):
    """Current officially supported dataset adapter for the standalone release."""

    name = "piebench"

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def export(self, output_dir: Path) -> list[StandardSample]:
        mapping_path = self.dataset_root / "mapping_file.json"
        image_root = self.dataset_root / "annotation_images"
        if not mapping_path.exists():
            raise FileNotFoundError(f"PIE-Bench mapping file not found: {mapping_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"PIE-Bench annotation_images not found: {image_root}")

        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("PIE-Bench mapping_file.json must be an object.")

        output_dir = Path(output_dir).expanduser().resolve()
        samples: list[StandardSample] = []
        for row_index, (record_id, record) in enumerate(sorted(payload.items(), key=lambda item: item[0])):
            image_path = image_root / str(record["image_path"])
            if not image_path.exists():
                continue
            source_prompt = str(record.get("original_prompt") or record.get("source_prompt") or "").strip()
            editing_prompt = str(record.get("editing_prompt") or record.get("target_prompt") or "").strip()
            editing_instruction = str(record.get("editing_instruction") or editing_prompt).strip()
            target_prompt = editing_prompt.replace("[", "").replace("]", "")

            sample_id = f"piebench_{row_index:06d}"
            sample_dir = output_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            source_image_path = sample_dir / "source.png"
            source_image = Image.open(image_path).convert("RGB")
            source_image.save(source_image_path)

            mask_path: Path | None = None
            rle = record.get("mask")
            if isinstance(rle, list) and rle:
                gt_mask = decode_piebench_rle(
                    [int(value) for value in rle],
                    height=source_image.height,
                    width=source_image.width,
                )
                mask_path = sample_dir / "gt_mask.png"
                Image.fromarray(np.uint8(gt_mask * 255)).save(mask_path)

            sample = StandardSample(
                sample_id=sample_id,
                source_image_path=source_image_path,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                metadata=SampleMetadata(
                    dataset=self.name,
                    record_id=str(record_id),
                    edit_instruction=editing_instruction,
                    extras={
                        "source_dataset_root": str(self.dataset_root),
                        "editing_type_id": record.get("editing_type_id"),
                        "blended_word": record.get("blended_word"),
                        "has_gt_mask": mask_path is not None,
                    },
                ),
                mask_path=mask_path,
            )
            samples.append(sample)

        self.write_standard_samples(samples, output_dir)
        manifest_path = output_dir / "manifest.json"
        write_json(manifest_path, {"samples": [str((output_dir / sample.sample_id / "sample.json").resolve()) for sample in samples]})
        return samples
