"""Tự động tạo Project và Tasks trên CVAT thông qua REST API.
Tạo Task gold-manual (4 ảnh) và Task batch-review (24 ảnh), sau đó nạp sẵn 12 bẫy (pre_traps_import.coco.json).
"""
import json
import os
import sys
import time
from pathlib import Path
import requests

CVAT_URL = "http://localhost:8080"
TOKEN = "898ec190532c2e627261958df5c3c4b0c840c120"
HEADERS = {"Authorization": f"Token {TOKEN}"}

ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = ROOT / "cvat" / "labels_generated.json"
GOLD_DIR = ROOT / "work" / "tasks" / "gold"
REVIEW_DIR = ROOT / "work" / "tasks" / "review"
PRE_TRAPS_FILE = ROOT / "submission" / "prelabel" / "pre_traps_import.coco.json"

def get_or_create_project(project_name="day08-2A202603069"):
    # Kiểm tra project đã tồn tại chưa
    res = requests.get(f"{CVAT_URL}/api/projects", headers=HEADERS)
    res.raise_for_status()
    for p in res.json().get("results", []):
        if p["name"] == project_name:
            print(f"[Project] Đã có project: {project_name} (ID: {p['id']})")
            return p["id"]

    # Đọc nhãn
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    payload = {
        "name": project_name,
        "labels": labels
    }
    print(f"[Project] Đang tạo project mới: {project_name}...")
    res = requests.post(f"{CVAT_URL}/api/projects", headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        print(f"Lỗi tạo project: {res.status_code} - {res.text}")
        sys.exit(1)
    p_id = res.json()["id"]
    print(f"[Project] Tạo thành công: ID {p_id}")
    return p_id

def wait_task_data(task_id, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{CVAT_URL}/api/tasks/{task_id}/status", headers=HEADERS)
        if r.status_code == 200:
            st = r.json().get("state")
            if st in ("Completed", "Finished"):
                return True
            elif st == "Failed":
                print(f"Lỗi nạp dữ liệu cho Task {task_id}: {r.json()}")
                return False
        time.sleep(1)
    return False

def create_task(name, project_id, image_dir):
    # Kiểm tra xem task đã tồn tại trong project chưa
    res = requests.get(f"{CVAT_URL}/api/tasks?project_id={project_id}", headers=HEADERS)
    res.raise_for_status()
    for t in res.json().get("results", []):
        if t["name"] == name:
            print(f"[Task] Đã có task: {name} (ID: {t['id']})")
            return t["id"]

    print(f"[Task] Đang tạo task: {name}...")
    payload = {
        "name": name,
        "project_id": project_id
    }
    res = requests.post(f"{CVAT_URL}/api/tasks", headers=HEADERS, json=payload)
    if res.status_code not in (200, 201):
        print(f"Lỗi tạo task: {res.status_code} - {res.text}")
        sys.exit(1)
    task_id = res.json()["id"]

    # Upload ảnh
    img_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    print(f"[Task {task_id}] Đang tải lên {len(img_files)} ảnh từ {image_dir.name}...")

    files = []
    opened_files = []
    try:
        for idx, img_path in enumerate(img_files):
            f = open(img_path, "rb")
            opened_files.append(f)
            files.append((f"client_files[{idx}]", (img_path.name, f, "image/jpeg")))

        data_payload = {
            "image_quality": "70",
            "use_zip_chunks": "true"
        }
        res_data = requests.post(
            f"{CVAT_URL}/api/tasks/{task_id}/data",
            headers=HEADERS,
            data=data_payload,
            files=files
        )
        if res_data.status_code not in (200, 202):
            print(f"Lỗi nạp ảnh cho Task {task_id}: {res_data.status_code} - {res_data.text}")
            sys.exit(1)
    finally:
        for f in opened_files:
            f.close()

    print(f"[Task {task_id}] Đang chờ CVAT xử lý ảnh...")
    if not wait_task_data(task_id):
        print(f"[Task {task_id}] Quá thời gian xử lý ảnh!")
        sys.exit(1)

    print(f"[Task {task_id}] Tạo và nạp ảnh thành công!")
    return task_id

def upload_annotations(task_id, anno_file):
    print(f"[Task {task_id}] Đang nạp annotations từ {anno_file.name} (COCO 1.0)...")
    url = f"{CVAT_URL}/api/tasks/{task_id}/annotations?format=COCO%201.0"
    with open(anno_file, "rb") as f:
        res = requests.post(url, headers=HEADERS, files={"annotation_file": (anno_file.name, f, "application/json")})
    
    if res.status_code in (200, 201, 202):
        print(f"[Task {task_id}] Yêu cầu nạp annotations đã được tiếp nhận ({res.status_code}).")
        # Đợi xử lý nếu có rq_id
        try:
            rq_id = res.json().get("rq_id")
            if rq_id:
                for _ in range(30):
                    time.sleep(1)
                    r = requests.get(f"{CVAT_URL}/api/requests/{rq_id}", headers=HEADERS)
                    if r.status_code == 200 and r.json().get("status") in ("finished", "completed"):
                        break
        except Exception:
            pass
        print(f"[Task {task_id}] Nạp annotations hoàn tất!")
    else:
        print(f"[Task {task_id}] Cảnh báo khi nạp annotations: {res.status_code} - {res.text}")

def get_task_jobs(task_id):
    res = requests.get(f"{CVAT_URL}/api/jobs?task_id={task_id}", headers=HEADERS)
    if res.status_code == 200:
        results = res.json().get("results", [])
        if results:
            return results[0]["id"]
    return None

def main():
    print("=== TỰ ĐỘNG THIẾT LẬP TASK CVAT CHO DAY 08 ===")
    project_id = get_or_create_project("day08-2A202603069")

    # 1. Tạo task gold-manual (4 ảnh)
    gold_task_id = create_task("gold-manual", project_id, GOLD_DIR)
    gold_job_id = get_task_jobs(gold_task_id)

    # 2. Tạo task batch-review (24 ảnh)
    batch_task_id = create_task("batch-review", project_id, REVIEW_DIR)
    batch_job_id = get_task_jobs(batch_task_id)

    # 3. Nạp annotations 12 bẫy vào batch-review
    if PRE_TRAPS_FILE.exists():
        upload_annotations(batch_task_id, PRE_TRAPS_FILE)

    print("\n================ THÀNH CÔNG RỰC RỠ ================")
    print(f"Project ID: {project_id} (day08-2A202603069)")
    print(f"1. Task Gold (4 ảnh gán tay):")
    print(f"   - Task URL: {CVAT_URL}/tasks/{gold_task_id}")
    if gold_job_id:
        print(f"   - Link vào gán nhãn ngay: {CVAT_URL}/tasks/{gold_task_id}/jobs/{gold_job_id}")
    
    print(f"\n2. Task Batch Review (24 ảnh + 12 bẫy đã nạp sẵn):")
    print(f"   - Task URL: {CVAT_URL}/tasks/{batch_task_id}")
    if batch_job_id:
        print(f"   - Link vào soi bẫy ngay: {CVAT_URL}/tasks/{batch_task_id}/jobs/{batch_job_id}")
    print("====================================================")

if __name__ == "__main__":
    main()
