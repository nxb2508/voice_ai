from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
    Response,
    Depends,
    Query,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import tempfile
from pydantic import BaseModel
from typing import List, Optional, Literal
from so_vits_svc_fork.inference.main import infer
import os
from fastapi.middleware.cors import CORSMiddleware
import uuid
from gtts import gTTS
from docx import Document
import io
import logging
from logging import getLogger
from so_vits_svc_fork.preprocessing.preprocess_split import preprocess_split
from so_vits_svc_fork.preprocessing.preprocess_resample import preprocess_resample
from so_vits_svc_fork.preprocessing.preprocess_flist_config import (
    CONFIG_TEMPLATE_DIR,
    preprocess_config,
)
from so_vits_svc_fork.preprocessing.preprocess_hubert_f0 import preprocess_hubert_f0
from so_vits_svc_fork.train import train
import shutil
import shutil
import asyncio
import aiofiles
import uvicorn
import re
import firebase_admin
from firebase_admin import credentials, firestore
from concurrent.futures import ThreadPoolExecutor
import json

BASE_DIR = Path(__file__).resolve().parent


cred = credentials.Certificate(BASE_DIR / "APIkey.json")
firebase_admin.initialize_app(cred)
db_firestore = firestore.client()


section_storage_path = BASE_DIR / "files"
train_model_path = BASE_DIR / "trainmodel"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=20)
# Tạo FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Định nghĩa schema cho phản hồi
class ModelResponse(BaseModel):
    id_model: str = None
    model_name: str
    model_path: str
    config_path: str
    cluster_model_path: str
    category: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class ModelCreate(BaseModel):
    model_name: str
    model_path: str
    config_path: str
    cluster_model_path: str
    category: str


class TextToSpeechRequest(BaseModel):
    text: str
    locate: str = "vi"


class TextToSpeechAndInferRequest(BaseModel):
    text: str
    locate: str = "vi"
    model_id: str


def get_latest_model_path(log_dir):
    max_epoch = -1
    model_path = None

    for filename in os.listdir(log_dir):

        match = re.match(r"G_(\d+)\.pth", filename)
        if match:
            epoch = int(match.group(1))

            if epoch > max_epoch:
                max_epoch = epoch
                model_path = Path(log_dir) / filename

    return str(model_path).replace("\\", "/") if model_path else None


async def save_uploaded_file(uploaded_file, destination: Path):
    try:
        with destination.open("wb") as dest_file:
            while True:
                chunk = await asyncio.to_thread(uploaded_file.read, 1024)
                if not chunk:
                    break
                dest_file.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Error saving uploaded file: {e}")


def update_config(file_path):

    with open(file_path, "r") as file:
        config = json.load(file)

    config["train"]["epochs"] = 3000
    config["train"]["batch_size"] = 1
    config["train"]["log_interval"] = 200
    config["train"]["eval_interval"] = 800
    config["train"]["learning_rate"] = 0.0015

    with open(file_path, "w") as file:
        json.dump(config, file, indent=4)
    print("update thành công")


def pre_split(
    input_dir: str,
    output_dir: str,
    sr: int,
    max_length: float = 5.0,
    top_db: int = 30,
    frame_seconds: float = 0.5,
    hop_seconds: float = 0.1,
    n_jobs: int = -1,
) -> None:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_split(
        input_dir=input_dir,
        output_dir=output_dir,
        sr=sr,
        max_length=max_length,
        top_db=top_db,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        n_jobs=n_jobs,
    )


def process_audio_files(
    input_dir: str,
    output_dir: str,
    sampling_rate: int = 16000,
    top_db: int = 30,
    frame_seconds: float = 0.1,
    hop_seconds: float = 0.05,
    n_jobs: int = -1,
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Tạo thư mục output nếu chưa tồn tại
    output_path.mkdir(parents=True, exist_ok=True)

    # Gọi hàm preprocess_resample để xử lý các tệp âm thanh
    preprocess_resample(
        input_dir=input_path,
        output_dir=output_path,
        sampling_rate=sampling_rate,
        n_jobs=n_jobs,
        top_db=top_db,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
    )


def pre_config(
    input_dir: Path,
    filelist_path: Path,
    config_path: Path,
    config_type: str,
):

    input_dir = Path(input_dir)
    filelist_path = Path(filelist_path)
    config_path = Path(config_path)
    preprocess_config(
        input_dir=input_dir,
        train_list_path=filelist_path / "train.txt",
        val_list_path=filelist_path / "val.txt",
        test_list_path=filelist_path / "test.txt",
        config_path=config_path,
        config_name=config_type,
    )

    print("Configuration processing completed.")


def pre_hubert(
    input_dir: Path,
    config_path: Path,
    n_jobs: bool,
    force_rebuild: bool,
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
) -> None:
    from so_vits_svc_fork.preprocessing.preprocess_hubert_f0 import preprocess_hubert_f0

    input_dir = Path(input_dir)
    config_path = Path(config_path)
    preprocess_hubert_f0(
        input_dir=input_dir,
        config_path=config_path,
        n_jobs=n_jobs,
        force_rebuild=force_rebuild,
        f0_method=f0_method,
    )


def train(
    config_path: Path,
    model_path: Path,
    tensorboard: bool = False,
    reset_optimizer: bool = False,
):
    config_path = Path(config_path)
    model_path = Path(model_path)

    if tensorboard:
        import webbrowser
        from tensorboard import program

        getLogger("tensorboard").setLevel(30)
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", model_path.as_posix()])
        url = tb.launch()
        webbrowser.open(url)

    # Gọi hàm huấn luyện thực tế từ module khác, nếu cần
    from so_vits_svc_fork.train import train  # Chỉ gọi một lần từ đây

    train(
        config_path=config_path, model_path=model_path, reset_optimizer=reset_optimizer
    )


@app.get("/models", response_model=List[ModelResponse])
async def get_models():
    try:
        models_ref = db_firestore.collection("models")
        docs = models_ref.stream()

        models = []
        for doc in docs:
            data = doc.to_dict()
            model = ModelResponse(
                id_model=doc.id,
                model_name=data["model_name"],
                model_path=data["model_path"],
                config_path=data["config_path"],
                cluster_model_path=data.get("cluster_model_path", ""),
                category=data.get("category", ""),
            )
            models.append(model)

        return models
    except Exception as e:
        logger.error(f"Lỗi khi lấy models: {e}")
        raise HTTPException(status_code=500, detail="Không thể lấy danh sách models.")


# Thêm model
@app.post("/models/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(model: ModelCreate):

    model_id = str(uuid.uuid4())

    model_ref = db_firestore.collection("models").document(model_id)
    model_ref.set(
        {
            "model_name": model.model_name,
            "model_path": model.model_path,
            "config_path": model.config_path,
            "cluster_model_path": model.cluster_model_path,
            "category": model.category,
        }
    )

    return ModelResponse(
        id_model=model_id,
        model_name=model.model_name,
        model_path=model.model_path,
        config_path=model.config_path,
        cluster_model_path=model.cluster_model_path,
        category=model.category,
    )


# Xóa một model theo ID
@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    try:
        db_firestore.collection("models").document(str(model_id)).delete()
        return {"message": "Xóa model thành công."}
    except Exception as e:
        logger.error(f"Lỗi khi xóa model: {e}")
        raise HTTPException(status_code=500, detail="Không thể xóa model.")


# API TFTS và Infer
@app.post("/text-file-to-speech-and-infer/")
async def text_file_to_speech_and_infer(
    file: UploadFile = File(...),
    locate: str = Form("vi"),
    model_id: str = Form(...),
):
    if file.filename.endswith(".txt"):
        # Đọc nội dung từ tệp văn bản
        os.makedirs(section_storage_path, exist_ok=True)
        os.makedirs(train_model_path, exist_ok=True)

        content = await file.read()
        text = content.decode("utf-8")

        try:
            model_doc = db_firestore.collection("models").document(str(model_id)).get()
            if not model_doc.exists:
                raise HTTPException(status_code=400, detail="Model not found")

            model_info = model_doc.to_dict()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Lỗi khi lấy thông tin model: {e}"
            )

        model_path = BASE_DIR / model_info["model_path"]
        config_path = BASE_DIR / model_info["config_path"]
        cluster_model_path = BASE_DIR / model_info["cluster_model_path"]

        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

        try:
            tts = gTTS(text=text, lang=locate)

            await asyncio.to_thread(tts.save, audio_file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        output_audio_path = os.path.join(
            section_storage_path, f"{section_id}_processed.wav"
        )

        # Gọi hàm infer với file âm thanh vừa tạo ra
        try:
            await asyncio.to_thread(
                infer,
                input_path=audio_file_path,
                output_path=Path(output_audio_path),
                model_path=Path(model_path),
                config_path=Path(config_path),
                speaker=0,
                cluster_model_path=Path(cluster_model_path),
                transpose=0,
                auto_predict_f0=True,
                cluster_infer_ratio=0.0,
                noise_scale=0.4,
                f0_method="dio",
                db_thresh=-40,
                pad_seconds=0.5,
                chunk_seconds=0.5,
                absolute_thresh=False,
                max_chunk_seconds=40,
            )
            # print(f"Audio đã được xử lý và lưu tại: {output_audio_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing audio: {str(e)}"
            )

        return FileResponse(
            output_audio_path, media_type="audio/wav", filename="output.wav"
        )
    elif file.filename.endswith(".docx"):
        os.makedirs(section_storage_path, exist_ok=True)
        os.makedirs(train_model_path, exist_ok=True)
        # Đọc nội dung từ tệp Word
        content = await file.read()
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        # Lấy model_path và config_path từ cơ sở dữ liệu
        try:
            model_doc = db_firestore.collection("models").document(str(model_id)).get()
            if not model_doc.exists:
                raise HTTPException(status_code=400, detail="Model not found")

            model_info = model_doc.to_dict()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Lỗi khi lấy thông tin model: {e}"
            )

        model_path = BASE_DIR / model_info["model_path"]
        config_path = BASE_DIR / model_info["config_path"]
        cluster_model_path = BASE_DIR / model_info["cluster_model_path"]

        # Tạo tên file âm thanh ngẫu nhiên
        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

        # Chuyển đổi văn bản thành giọng nói
        try:
            tts = gTTS(text=text, lang=locate)
            # Lưu file âm thanh trong một luồng riêng biệt
            await asyncio.to_thread(tts.save, audio_file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        output_audio_path = os.path.join(
            section_storage_path, f"{section_id}_processed.wav"
        )

        # Gọi hàm infer với file âm thanh vừa tạo ra
        try:
            await asyncio.to_thread(
                infer,
                input_path=audio_file_path,
                output_path=Path(output_audio_path),
                model_path=Path(model_path),
                config_path=Path(config_path),
                speaker=0,
                cluster_model_path=Path(cluster_model_path),
                transpose=0,
                auto_predict_f0=True,
                cluster_infer_ratio=0.0,
                noise_scale=0.4,
                f0_method="dio",
                db_thresh=-40,
                pad_seconds=0.5,
                chunk_seconds=0.5,
                absolute_thresh=False,
                max_chunk_seconds=40,
            )
            # print(f"Audio đã được xử lý và lưu tại: {output_audio_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing audio: {str(e)}"
            )

        return FileResponse(
            output_audio_path, media_type="audio/wav", filename="output.wav"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")


@app.post("/text-to-speech-and-infer/")
async def text_to_speech_and_process(request: TextToSpeechAndInferRequest):
    model_id = request.model_id
    text = request.text
    locate = request.locate

    os.makedirs(section_storage_path, exist_ok=True)
    os.makedirs(train_model_path, exist_ok=True)
    # Kiểm tra thông tin mô hình trong cơ sở dữ liệu
    try:
        model_doc = db_firestore.collection("models").document(str(model_id)).get()
        if not model_doc.exists:
            raise HTTPException(status_code=400, detail="Model not found")

        model_info = model_doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin model: {e}")

    # Lấy thông tin đường dẫn mô hình và cấu hình từ cơ sở dữ liệu
    model_path = BASE_DIR / model_info["model_path"]
    config_path = BASE_DIR / model_info["config_path"]
    cluster_model_path = BASE_DIR / model_info["cluster_model_path"]

    # Tạo tên file ngẫu nhiên
    section_id = str(uuid.uuid4())
    section_cloned_file_path = os.path.join(section_storage_path, section_id + ".wav")

    # Sử dụng gTTS để chuyển đổi văn bản thành giọng nói
    try:
        tts = gTTS(text=text, lang=locate)

        await asyncio.to_thread(tts.save, section_cloned_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    # Gọi hàm infer để xử lý âm thanh
    output_audio_path = os.path.join(
        section_storage_path, f"{section_id}_processed.wav"
    )
    try:
        await asyncio.to_thread(
            infer,
            input_path=section_cloned_file_path,
            output_path=Path(output_audio_path),
            model_path=Path(model_path),
            config_path=Path(config_path),
            speaker=0,
            cluster_model_path=Path(cluster_model_path),
            transpose=0,
            auto_predict_f0=True,
            cluster_infer_ratio=0.0,
            noise_scale=0.4,
            f0_method="dio",
            db_thresh=-40,
            pad_seconds=0.5,
            chunk_seconds=0.5,
            absolute_thresh=False,
            max_chunk_seconds=40,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    return FileResponse(
        output_audio_path, media_type="audio/wav", filename="output.wav"
    )


@app.post("/infer-audio/", response_class=FileResponse)
async def upload_and_infer(
    file: UploadFile = File(...),
    model_id: str = Form(...),
):

    # Kiểm tra thông tin mô hình trong cơ sở dữ liệu
    try:
        model_doc = db_firestore.collection("models").document(str(model_id)).get()
        if not model_doc.exists:
            raise HTTPException(status_code=400, detail="Model not found")

        model_info = model_doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy thông tin model: {e}")

    os.makedirs(section_storage_path, exist_ok=True)
    os.makedirs(train_model_path, exist_ok=True)
    input_id = str(uuid.uuid4())
    output_id = str(uuid.uuid4())

    input_file_path = os.path.join(section_storage_path, f"{input_id}.wav")
    output_file_path = os.path.join(section_storage_path, f"{output_id}_processed.wav")

    cluster_model_path = BASE_DIR / model_info["cluster_model_path"]
    # Lưu tệp âm thanh được tải lên bằng aiofiles
    async with aiofiles.open(input_file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    # Gọi hàm infer để xử lý âm thanh
    await asyncio.to_thread(
        infer,
        input_path=input_file_path,
        output_path=Path(output_file_path),
        model_path=Path(BASE_DIR / model_info["model_path"]),
        config_path=Path(BASE_DIR / model_info["config_path"]),
        speaker=0,
        cluster_model_path=Path(cluster_model_path),
        transpose=0,
        auto_predict_f0=True,
        cluster_infer_ratio=0.0,
        noise_scale=0.4,
        f0_method="dio",
        db_thresh=-40,
        pad_seconds=0.5,
        chunk_seconds=0.5,
        absolute_thresh=False,
        max_chunk_seconds=40,
    )

    print(f"Audio đã được xử lý và lưu tại: {output_file_path}")
    return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")


@app.post("/train-model/")
async def process_audio(
    name: str = Form(...), file: UploadFile = File(...), f0_method: str = Form(...)
):
    # Tạo thư mục tạm để lưu file âm thanh
    suid = str(uuid.uuid4())[:8]
    file_name = name + suid
    input_dir = BASE_DIR / f"audio_data/{file_name}"
    output_dir = BASE_DIR / f"trainmodel/{file_name}"
    input_train = BASE_DIR / f"trainmodel/{file_name}/dataset/44k"
    output_split = BASE_DIR / f"trainmodel/{file_name}/dataset_raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    temp_file_path = Path(input_dir) / file.filename

    # Lưu tệp âm thanh vào thư mục
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    await save_uploaded_file(file.file, temp_file_path)

    # Chạy lần lượt các hàm xử lý
    try:
        # Bước 1: Pre-split
        await asyncio.to_thread(
            pre_split,
            input_dir=input_dir,
            output_dir=os.path.join(output_dir, f"dataset_raw/{name}"),
            sr=22050,
        )
        # Bước 2: Process Audio Files
        await asyncio.to_thread(
            process_audio_files,
            input_dir=output_split,
            output_dir=input_train,
            sampling_rate=16000,
        )

        # Bước 3: Pre-config
        config_dir = os.path.join(output_dir, "configs/44k/config.json")
        filelist_dir = os.path.join(output_dir, "filelists/44k")
        config_type = "so-vits-svc-4.0v1"
        await asyncio.to_thread(
            pre_config,
            input_dir=input_train,
            filelist_path=filelist_dir,
            config_path=config_dir,
            config_type=config_type,
        )
        update_config(config_dir)
        # Bước 4: Pre-hubert
        await asyncio.to_thread(
            pre_hubert,
            input_dir=input_train,
            config_path=config_dir,
            n_jobs=None,
            force_rebuild=True,
            f0_method=f0_method,
        )

        # Bước 5: Train model
        model_dir = os.path.join(output_dir, "logs/44k")
        await asyncio.to_thread(
            train,
            config_path=config_dir,
            model_path=model_dir,
            tensorboard=False,
            reset_optimizer=False,
        )
        latest_model_path = get_latest_model_path(model_dir)
        config_path_relative = Path(config_dir).relative_to(BASE_DIR).as_posix()
        latest_model_path_relative = (
            Path(latest_model_path).relative_to(BASE_DIR).as_posix()
        )

        # Lưu thông tin model vào Firestore
        model_id = str(uuid.uuid4())
        model_ref = db_firestore.collection("models").document(model_id)
        model_data = {
            "model_name": file_name,
            "model_path": latest_model_path_relative,
            "config_path": config_path_relative,
            "cluster_model_path": "None",
            "category": "user_train",
        }
        model_ref.set(model_data)
        return {
            "message": "Audio processing completed successfully!",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)