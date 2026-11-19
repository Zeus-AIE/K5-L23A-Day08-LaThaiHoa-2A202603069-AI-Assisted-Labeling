"""Chạy trực tiếp Meta SAM 3 (D:\\App\\sam3.pt) trên GPU RTX 3060 theo hai chế độ:
1. Interactive Click Prompt (model.predict_inst) -> tuktuk_imgXXX_sam3.jpg
2. Text-Grounded Concept Prompt (processor.set_text_prompt) -> tuktuk_imgXXX_sam3_text.jpg
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

SAM3_REPO = Path(r"D:\App\sam3_repo")
CHECKPOINT_PATH = Path(r"D:\App\sam3.pt")

if str(SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submission" / "masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    {
        "name": "tuktuk_img013_sam3",
        "image_file": ROOT / "data" / "images" / "img_013.jpg",
        "pos_points": [[400, 480], [600, 110]],
        "neg_points": [[720, 300]],
        "text_prompt": "tuk-tuk",
        "desc": "Tuk-tuk tiền cảnh có người lái (img_013)",
    },
    {
        "name": "tuktuk_img004_sam3",
        "image_file": ROOT / "data" / "images" / "img_004.jpg",
        "pos_points": [[650, 410], [550, 350]],
        "neg_points": [[400, 300]],
        "text_prompt": "tuk-tuk",
        "desc": "Tuk-tuk vàng trong bãi đỗ (img_004)",
    },
    {
        "name": "tuktuk_img017_sam3",
        "image_file": ROOT / "data" / "images" / "img_017.jpg",
        "pos_points": [[520, 480], [800, 460], [630, 345]],
        "neg_points": [[550, 250]],
        "text_prompt": "tuk-tuk",
        "desc": "Ba xe tuk-tuk đậu dọc lề đường (img_017)",
    },
]

PALETTE = [
    (255, 140, 0),
    (0, 200, 255),
    (50, 220, 100),
    (220, 80, 220),
]


def draw_mask_overlay(
    img_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.50,
) -> np.ndarray:
    """Phủ mask bán trong suốt và vẽ contour viền trắng lên ảnh BGR."""
    out = img_bgr.copy()
    if mask_bool.any():
        out[mask_bool] = (
            out[mask_bool].astype(np.float32) * (1.0 - alpha)
            + np.array(color_bgr, dtype=np.float32) * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint SAM 3 tại {CHECKPOINT_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CHẠY META SAM 3 CHÍNH THỨC ({device.upper()}) ===")
    print(f"Đang nạp checkpoint: {CHECKPOINT_PATH}...")

    model = build_sam3_image_model(
        checkpoint_path=str(CHECKPOINT_PATH),
        load_from_HF=False,
        device=device,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model=model, device=device, confidence_threshold=0.35)

    for item in TARGETS:
        print(f"\n[Xử lý] {item['desc']}...")
        img_bgr = cv2.imread(str(item["image_file"]))
        h, w = img_bgr.shape[:2]
        pil_img = Image.open(item["image_file"]).convert("RGB")

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            state = processor.set_image(pil_img)

            # --- Chế độ A: Interactive Click (predict_inst) ---
            pts = item["pos_points"] + item["neg_points"]
            lbls = [1] * len(item["pos_points"]) + [0] * len(item["neg_points"])
            pts_np = np.array(pts, dtype=np.float32)
            lbls_np = np.array(lbls, dtype=np.int32)

            masks, scores, _ = model.predict_inst(
                state,
                point_coords=pts_np,
                point_labels=lbls_np,
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_mask = np.asarray(masks[best_idx]).squeeze() > 0
            if best_mask.shape != (h, w):
                best_mask = (
                    cv2.resize(
                        best_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                    > 0
                )
            click_area = int(best_mask.sum())

            click_vis = draw_mask_overlay(img_bgr, best_mask, (255, 140, 0), alpha=0.52)
            for pt in item["pos_points"]:
                cv2.circle(click_vis, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), -1)
                cv2.circle(click_vis, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)
            for pt in item["neg_points"]:
                cv2.circle(click_vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
                cv2.circle(click_vis, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)

            cv2.putText(
                click_vis,
                f"Meta SAM 3 Click (Conf: {best_score:.3f}, Area: {click_area:,} px)",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            click_out = OUT_DIR / f"{item['name']}.jpg"
            cv2.imwrite(str(click_out), click_vis)
            print(
                f"  -> [Click] {click_out.name}: Conf={best_score:.3f}, Area={click_area:,} px"
            )

            # --- Chế độ B: Text-Grounded Concept Prompt (set_text_prompt) ---
            used_prompt = item["text_prompt"]
            text_out = processor.set_text_prompt(prompt=used_prompt, state=state)
            raw_masks = text_out.get("masks", [])
            if len(raw_masks) == 0:
                used_prompt = "three-wheeled vehicle"
                text_out = processor.set_text_prompt(prompt=used_prompt, state=state)
                raw_masks = text_out.get("masks", [])

            raw_boxes = text_out.get("boxes", [])
            raw_scores = text_out.get("scores", [])

            text_vis = img_bgr.copy()
            total_text_area = 0
            inst_count = len(raw_masks)
            for idx in range(inst_count):
                m = raw_masks[idx]
                m_np = (
                    m.squeeze().detach().cpu().numpy() > 0
                    if torch.is_tensor(m)
                    else np.asarray(m).squeeze() > 0
                )
                if m_np.shape != (h, w):
                    m_np = (
                        cv2.resize(
                            m_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                        )
                        > 0
                    )
                total_text_area += int(m_np.sum())
                color = PALETTE[idx % len(PALETTE)]
                text_vis = draw_mask_overlay(text_vis, m_np, color, alpha=0.48)

                sc = (
                    float(raw_scores[idx].item())
                    if idx < len(raw_scores) and torch.is_tensor(raw_scores[idx])
                    else float(raw_scores[idx])
                    if idx < len(raw_scores)
                    else 0.0
                )
                if idx < len(raw_boxes):
                    b = raw_boxes[idx]
                    b_np = (
                        b.detach().cpu().numpy()
                        if torch.is_tensor(b)
                        else np.asarray(b)
                    )
                    x1, y1, x2, y2 = [int(v) for v in b_np[:4]]
                    cv2.rectangle(text_vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        text_vis,
                        f"{used_prompt} {sc:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                text_vis,
                f"Meta SAM 3 Text ('{used_prompt}': {inst_count} inst, Area: {total_text_area:,} px)",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            text_out_path = OUT_DIR / f"{item['name']}_text.jpg"
            cv2.imwrite(str(text_out_path), text_vis)
            print(
                f"  -> [Text]  {text_out_path.name}: Prompt='{used_prompt}', Inst={inst_count}, Area={total_text_area:,} px"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nHoàn tất chạy SAM 3 ở cả 2 chế độ!")


if __name__ == "__main__":
    run()
