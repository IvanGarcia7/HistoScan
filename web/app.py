import os
import sys
import uuid
import shutil
import json
import subprocess
import threading
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="HistoScan AI | Enterprise Engine")

# Configuration paths
ROOT_DIR = "/app" 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
SEG_OUT_DIR = os.path.join(OUTPUT_DIR, "seg")
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data", "segmentation")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEG_OUT_DIR, exist_ok=True)


app.mount("/static", StaticFiles(directory=os.path.join(CURRENT_DIR, "static")), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=os.path.join(CURRENT_DIR, "templates"))

jobs = {}
# Global training status
training_status = {
    "is_training": False, 
    "epoch": 0, 
    "loss": 0, 
    "dice": 0, 
    "iou": 0, 
    "history": []
}
# Track current training directory for status updates
current_training_dir = SEG_OUT_DIR

# ================================================================
# WSI MODULE
# ================================================================
def run_wsi_job(job_id, filename, target_tiles, filter_empty):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 5
        
        slide_path = os.path.join(UPLOAD_DIR, filename)
        job_out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(os.path.join(job_out_dir, "tiles_sample"), exist_ok=True)

        wsi_script_dir = os.path.join(ROOT_DIR, "wsi")
        
        cmd = [
            sys.executable, "wsi_tiler.py", 
            "--slide", slide_path,
            "--out", job_out_dir,
            "--max-tiles", str(target_tiles),
        ]
        if filter_empty: cmd.append("--filter-empty")

        print(f"Exec WSI: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=wsi_script_dir) 

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
    except Exception as e:
        print(f"Error WSI: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

# ================================================================
# SEGMENTATION MODULE
# ================================================================
def run_training_task(epochs, batch_size, patience, model_arch, model_name, demo, train_dir, val_dir):
    global current_training_dir
    try:
        training_status["is_training"] = True
        training_status["history"] = []
        
        # 1. Create model-specific output directory
        folder_name = model_name.replace(".ckpt", "")
        model_specific_dir = os.path.join(SEG_OUT_DIR, folder_name)
        os.makedirs(model_specific_dir, exist_ok=True)
        
        # Update global path for /status endpoint
        current_training_dir = model_specific_dir
        
        # Clean old history if exists
        history_path = os.path.join(model_specific_dir, "history.json")
        if os.path.exists(history_path): os.remove(history_path)

        cmd = [
            sys.executable, "-m", "seg.train",
            "--train_dir", train_dir,
            "--val_dir", val_dir,
            "--out", model_specific_dir,  
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--model", model_arch,
            "--save_name", model_name
        ]
        if demo: cmd.append("--demo")
            
        print(f"Exec Training: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)
        
    except Exception as e:
        print(f"Error Train: {e}")
    finally:
        training_status["is_training"] = False

# ROUTES
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...), num_tiles: int = Form(50), filter_empty: bool = Form(False)):
    job_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    jobs[job_id] = {"status": "queued", "progress": 0, "filename": file.filename, "total_tiles_target": num_tiles}
    threading.Thread(target=run_wsi_job, args=(job_id, file.filename, num_tiles, filter_empty)).start()
    return JSONResponse({"job_id": job_id})

@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job: return {"status": "unknown"}
    if job["status"] == "running":
        try:
            out_dir = os.path.join(OUTPUT_DIR, job_id, "tiles_sample")
            if os.path.exists(out_dir):
                files = [f for f in os.listdir(out_dir) if f.endswith(('.jpg', '.png'))]
                target = job.get("total_tiles_target", 100)
                job["progress"] = min(int((len(files) / target) * 100), 99)
        except: pass
    return job

@app.get("/result/{job_id}", response_class=HTMLResponse)
def result(request: Request, job_id: str):
    out_dir = os.path.join(OUTPUT_DIR, job_id)
    pyramid_path = os.path.join(out_dir, "pyramid.json")
    if not os.path.exists(pyramid_path): return HTMLResponse("Processing...", status_code=202)
    with open(pyramid_path) as f: pyramid = json.load(f)
    base = pyramid['levels'][0]
    meta = {"megapixels": round((base['width'] * base['height']) / 1_000_000, 2), "aspect": round(base['width'] / base['height'], 2)}
    tiles = sorted(os.listdir(os.path.join(out_dir, "tiles_sample"))) if os.path.exists(os.path.join(out_dir, "tiles_sample")) else []
    return templates.TemplateResponse("result.html", {"request": request, "pyramid": pyramid, "tiles": tiles, "job_id": job_id, "meta": meta})

@app.get("/download_zip/{job_id}")
def download_zip(job_id: str):
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    zip_path = os.path.join(OUTPUT_DIR, f"{job_id}_results")
    shutil.make_archive(zip_path, 'zip', job_dir)
    return FileResponse(f"{zip_path}.zip", media_type='application/zip', filename=f"analysis_{job_id}.zip")

# --- BUSCAR ESTA FUNCIÓN Y ACTUALIZARLA ---
@app.get("/segmentation", response_class=HTMLResponse)
def segmentation_ui(request: Request):
    # 1. Search for trained models (.ckpt files)
    ckpts = []
    if os.path.exists(SEG_OUT_DIR):
        for root, dirs, files in os.walk(SEG_OUT_DIR):
            for file in files:
                if file.endswith(".ckpt"):
                    rel_path = os.path.relpath(os.path.join(root, file), SEG_OUT_DIR)
                    ckpts.append(rel_path.replace(os.sep, '/'))
    
    # 2. Search for WSI slides (.tiff, .svs, etc) in uploads folder
    slides = []
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith(('.tiff', '.tif', '.svs', '.ndpi')):
                slides.append(f)

    return templates.TemplateResponse("segmentation.html", {
        "request": request, 
        "checkpoints": ckpts,
        "slides": slides,
        "default_train": os.path.join(DEFAULT_DATA_DIR, "train"),
        "default_val": os.path.join(DEFAULT_DATA_DIR, "val")
    })

@app.post("/seg/train/")
async def start_training(
    epochs: int = Form(50), 
    batch_size: int = Form(8),
    patience: int = Form(10),
    model_arch: str = Form("unet"),
    model_name: str = Form("best.ckpt"),
    demo: bool = Form(False), 
    train_path: str = Form(...), 
    val_path: str = Form(...)
):
    if training_status["is_training"]: return JSONResponse({"error": "Training active"}, status_code=400)
    
    thread = threading.Thread(
        target=run_training_task, 
        args=(epochs, batch_size, patience, model_arch, model_name, demo, train_path, val_path)
    )
    thread.start()
    return JSONResponse({"status": "started"})

@app.get("/seg/status")
def get_training_status():
    # Read history.json from current training directory
    global current_training_dir
    history_path = os.path.join(current_training_dir, "history.json")
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                data = json.load(f)
                if data:
                    last = data[-1]
                    training_status.update(last)
                    training_status["history"] = data
        except: pass
    return JSONResponse(training_status)

@app.post("/seg/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".ckpt"): return JSONResponse({"error": "Invalid file"}, status_code=400)
    # Save to segmentation output directory
    file_path = os.path.join(SEG_OUT_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    return JSONResponse({"status": "uploaded"})

@app.post("/seg/predict/")
async def predict_segmentation(file: UploadFile = File(...), ckpt: str = Form(...)):
    # 1. Save original image
    temp_id = uuid.uuid4().hex
    input_filename = f"{temp_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)
    with open(input_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    
    # 2. Locate checkpoint and pred_test directory
    ckpt_full_path = os.path.join(SEG_OUT_DIR, ckpt)
    model_dir = os.path.dirname(ckpt_full_path)
    pred_test_dir = os.path.join(model_dir, "pred_test")
    
    # Ensure directory exists
    os.makedirs(pred_test_dir, exist_ok=True)
    
    output_filename = f"mask_{input_filename}"
    
    try:
        # Execute inference
        cmd = [sys.executable, "-m", "seg.predict", "--ckpt", ckpt_full_path, "--input", input_path, "--out", pred_test_dir]
        subprocess.run(cmd, check=True, cwd=ROOT_DIR)
        
        # Build web URLs
        rel_path = os.path.relpath(os.path.join(pred_test_dir, output_filename), OUTPUT_DIR)
        web_mask_url = f"/outputs/{rel_path.replace(os.sep, '/')}"
        
        return JSONResponse({"original": f"/uploads/{input_filename}", "mask": web_mask_url})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# WSI PREDICTION ROUTE
@app.post("/wsi/predict/")
async def wsi_predict(
    file: UploadFile = File(None),
    wsi_file: str = Form(None),
    ckpt_file: str = Form(...), 
    num_tiles: int = Form(16),
    tile_size: int = Form(1024),
    overlap: int = Form(64),
    smart_filter: str = Form("true"),
    demo_mode: str = Form("false")
):
    try:
        # Handle file upload vs selection
        if file and file.filename:
            wsi_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(wsi_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            selected_wsi = file.filename
        else:
            wsi_path = os.path.join(UPLOAD_DIR, wsi_file)

        if not os.path.exists(wsi_path):
            return JSONResponse({"error": "WSI file not found"}, status_code=400)

        ckpt_path = os.path.join(SEG_OUT_DIR, ckpt_file)
        job_id = uuid.uuid4().hex[:8]
        out_dir = os.path.join(OUTPUT_DIR, "wsi_preds", job_id)
        os.makedirs(out_dir, exist_ok=True)
        
        # Convert JS checkbox format to Python boolean
        is_smart = "true" if smart_filter == "true" else "false"
        is_demo = "true" if demo_mode == "true" else "false"

        cmd = [
            sys.executable, "-m", "seg.wsi_predict",
            "--slide", wsi_path,
            "--ckpt", ckpt_path,
            "--out", out_dir,
            "--num_tiles", str(num_tiles),
            "--tile_size", str(tile_size),
            "--overlap", str(overlap)
        ]
        
        # Add smart filter flag if enabled
        if is_smart == "true":
            cmd.append("--smart_filter")
        
        # Add demo mode flag
        if is_demo == "true": cmd.append("--demo_mode")

        print(f"Exec WSI Pro: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            cwd=ROOT_DIR, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Read script output for debugging
        for line in process.stdout:
            print(f"[WSI-SCRIPT] {line.strip()}")
            
        process.wait()
        
        if process.returncode != 0:
            err_msg = process.stderr.read()
            print(f"Error en script WSI: {err_msg}")
            return JSONResponse({"error": f"Script failed: {err_msg}"}, status_code=500)
        
        # Gather results
        # 1. Stitched reconstruction image
        stitched_url = f"/outputs/wsi_preds/{job_id}/full_reconstruction.jpg"
        
        # 2. Individual tile predictions
        tiles = []
        for f in sorted(os.listdir(out_dir)):
            if f.startswith("tile_") and f.endswith(".jpg"):
                tiles.append(f"/outputs/wsi_preds/{job_id}/{f}")

        return JSONResponse({
            "status": "success", 
            "stitched": stitched_url,
            "images": tiles
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/browse/")
def browse_files(path: str = ""):
    base_mount_point = "/app/data"
    target_path = path if path else base_mount_point
    
    if not target_path.startswith("/app"):
        target_path = base_mount_point

    while not os.path.exists(target_path) and target_path != "/app":
        target_path = os.path.dirname(target_path)
    
    if not os.path.exists(target_path):
        target_path = "/app"

    try:
        if os.path.isfile(target_path):
            target_path = os.path.dirname(target_path)

        entries = os.listdir(target_path)
        dirs = []
        for entry in entries:
            full_path = os.path.join(target_path, entry)
            if os.path.isdir(full_path):
                dirs.append({"name": entry, "path": full_path})
        
        dirs.sort(key=lambda x: x["name"])
        parent_dir = os.path.dirname(target_path)
        
        return JSONResponse({
            "current_path": target_path,
            "parent_path": parent_dir if parent_dir.startswith("/app") else target_path,
            "dirs": dirs
        })
    except Exception as e:
        return JSONResponse({"error": f"Error leyendo {target_path}: {str(e)}"}, status_code=500)