from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response, Depends
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import tempfile
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List
from so_vits_svc_fork.inference.main import infer
import os
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import NoResultFound
from sqlalchemy.future import select
import uuid
from gtts import gTTS
from docx import Document
import io
import logging

# Kết nối tới cơ sở dữ liệu MySQL
DATABASE_URL = "mysql+pymysql://root:g08022002@localhost/datn"  # Thay đổi username, password, và dbname của bạn
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

section_storage_path = "D:/files/"
os.makedirs(section_storage_path, exist_ok=True)


# Định nghĩa cơ sở dữ liệu
Base = declarative_base()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(Base):
    __tablename__ = "models"  # Tên bảng trong MySQL

    id_model = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), index=True)
    model_path = Column(String(200))
    config_path = Column(String(200))
    cluster_model_path = Column(String(200), nullable=True)


# Tạo FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Cho phép tất cả các nguồn. Bạn có thể điều chỉnh để chỉ cho phép các nguồn cụ thể.
    allow_credentials=True,
    allow_methods=[
        "*"
    ],  # Cho phép tất cả các phương thức HTTP (GET, POST, PUT, DELETE, v.v.).
    allow_headers=["*"],  # Cho phép tất cả các header.
)


# Định nghĩa schema cho phản hồi
class ModelResponse(BaseModel):
    id_model: int
    model_name: str
    model_path: str
    config_path: str
    cluster_model_path: str

    class Config:
        from_attributes = True
        protected_namespaces = ()


class TextToSpeechRequest(BaseModel):
    text: str
    locate: str = "vi"


class TextToSpeechAndInferRequest(BaseModel):
    text: str
    locate: str = "vi"
    model_id: int


# Tạo một phiên làm việc với cơ sở dữ liệu
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Route để lấy danh sách các mô hình từ MySQL
@app.get("/models", response_model=List[ModelResponse])
async def get_models(db: Session = Depends(get_db)):
    models = db.query(Model).all()
    return models


@app.post("/models/", response_model=ModelResponse)
async def create_model(
    model_name: str = Form(...),
    model_path: str = Form(...),
    config_path: str = Form(...),
    cluster_model_path: str = Form(None),
    db: Session = Depends(get_db),
):
    new_model = Model(
        model_name=model_name,
        model_path=model_path,
        config_path=config_path,
        cluster_model_path=cluster_model_path,
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


@app.put("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_name: str = Form(...),
    model_path: str = Form(...),
    config_path: str = Form(...),
    cluster_model_path: str = Form(None),
    db: Session = Depends(get_db),
):
    model_info = db.query(Model).filter(Model.id_model == model_id).first()
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info.model_name = model_name
    model_info.model_path = model_path
    model_info.config_path = config_path
    model_info.cluster_model_path = cluster_model_path

    db.commit()
    db.refresh(model_info)
    return model_info


@app.delete("/models/{model_id}", response_model=dict)
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    model_info = db.query(Model).filter(Model.id_model == model_id).first()
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(model_info)
    db.commit()
    return {"detail": "Model deleted successfully"}


# API TTS
@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    section_id = str(uuid.uuid4())
    audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

    # Generate audio file
    try:
        tts = gTTS(text=request.text, lang=request.locate)
        tts.save(audio_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    return FileResponse(audio_file_path, media_type="audio/wav", filename="output.wav")


@app.post("/text-file-to-speech/")
async def text_file_to_speech(
    file: UploadFile = File(...),
    locate: str = Form("vi"),
):
    if file.filename.endswith(".txt"):
        # Đọc nội dung từ tệp văn bản
        content = await file.read()
        text = content.decode("utf-8")  # Giải mã nội dung
        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

        # Chuyển đổi văn bản thành giọng nói
        try:
            tts = gTTS(text=text, lang=locate)
            tts.save(audio_file_path)
            # print(f"File âm thanh đã được lưu tại: {audio_file_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        return FileResponse(
            audio_file_path, media_type="audio/wav", filename="output.wav"
        )
    elif file.filename.endswith(".docx"):
        # Đọc nội dung từ tệp Word
        content = await file.read()
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")
        try:
            tts = gTTS(text=text, lang=locate)
            tts.save(audio_file_path)
            # print(f"File âm thanh đã được lưu tại: {audio_file_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        return FileResponse(
            audio_file_path, media_type="audio/wav", filename="output.wav"
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")


# API TFTS và Infer
@app.post("/text-file-to-speech-and-infer/")
async def text_file_to_speech_and_infer(
    file: UploadFile = File(...),
    locate: str = Form("vi"),
    model_id: int = Form(...),
    db: Session = Depends(get_db),
):
    if file.filename.endswith(".txt"):
        # Đọc nội dung từ tệp văn bản
        content = await file.read()
        text = content.decode("utf-8")  # Giải mã nội dung

        # Lấy model_path và config_path từ cơ sở dữ liệu
        try:
            model = db.execute(
                select(Model).where(Model.id_model == model_id)
            ).scalar_one()
            model_path = model.model_path
            config_path = model.config_path
        except NoResultFound:
            raise HTTPException(status_code=404, detail="Model not found")

        # Tạo tên file âm thanh ngẫu nhiên
        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

        # Chuyển đổi văn bản thành giọng nói
        try:
            tts = gTTS(text=text, lang=locate)
            tts.save(audio_file_path)
            # print(f"File âm thanh đã được lưu tại: {audio_file_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        output_audio_path = os.path.join(
            section_storage_path, f"{section_id}_processed.wav"
        )

        # Gọi hàm infer với file âm thanh vừa tạo ra
        try:
            infer(
                input_path=audio_file_path,  # Sử dụng tệp âm thanh đã lưu
                output_path=Path(output_audio_path),
                model_path=Path(model_path),
                config_path=Path(config_path),
                speaker=0,  # Thay đổi ID của speaker nếu cần
                cluster_model_path=None,
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
        # Đọc nội dung từ tệp Word
        content = await file.read()
        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        try:
            model = db.execute(
                select(Model).where(Model.id_model == model_id)
            ).scalar_one()
            model_path = model.model_path
            config_path = model.config_path
        except NoResultFound:
            raise HTTPException(status_code=404, detail="Model not found")

        # Tạo tên file âm thanh ngẫu nhiên
        section_id = str(uuid.uuid4())
        audio_file_path = os.path.join(section_storage_path, section_id + ".wav")

        # Chuyển đổi văn bản thành giọng nói
        try:
            tts = gTTS(text=text, lang=locate)
            tts.save(audio_file_path)
            # print(f"File âm thanh đã được lưu tại: {audio_file_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating audio: {str(e)}"
            )

        output_audio_path = os.path.join(
            section_storage_path, f"{section_id}_processed.wav"
        )

        # Gọi hàm infer với file âm thanh vừa tạo ra
        try:
            infer(
                input_path=audio_file_path,  # Sử dụng tệp âm thanh đã lưu
                output_path=Path(output_audio_path),
                model_path=Path(model_path),
                config_path=Path(config_path),
                speaker=0,  # Thay đổi ID của speaker nếu cần
                cluster_model_path=None,
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


# API TTS và infer
@app.post("/text-to-speech-and-infer/")
async def text_to_speech_and_process(request: TextToSpeechAndInferRequest):
    # Tạo phiên làm việc với cơ sở dữ liệu
    db: Session = SessionLocal()

    # Lấy model_path và config_path từ cơ sở dữ liệu
    try:
        model = db.execute(
            select(Model).where(Model.id_model == request.model_id)
        ).scalar_one()
        model_path = model.model_path
        config_path = model.config_path
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Model not found")

    # Tạo tên file ngẫu nhiên
    section_id = str(uuid.uuid4())
    section_cloned_file_path = os.path.join(section_storage_path, section_id + ".wav")

    # Sử dụng gTTS để chuyển đổi văn bản thành giọng nói
    try:
        tts = gTTS(text=request.text, lang=request.locate)
        tts.save(section_cloned_file_path)
        # print(f"File âm thanh đã được lưu tại: {section_cloned_file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

    # Gọi hàm infer với file âm thanh vừa tạo ra
    output_audio_path = os.path.join(
        section_storage_path, f"{section_id}_processed.wav"
    )
    try:
        infer(
            input_path=section_cloned_file_path,
            output_path=Path(output_audio_path),
            model_path=Path(model_path),
            config_path=Path(config_path),
            speaker=0,
            cluster_model_path=None,
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
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    return FileResponse(
        output_audio_path, media_type="audio/wav", filename="output.wav"
    )


# API infer audio
@app.post("/infer-audio/", response_class=FileResponse)
async def upload_and_infer(
    file: UploadFile = File(...),
    model_id: int = Form(...),
    db: Session = Depends(get_db),
):
    # Kiểm tra thông tin mô hình trong cơ sở dữ liệu
    model_info = db.query(Model).filter(Model.id_model == model_id).first()
    if not model_info:
        raise HTTPException(status_code=400, detail="Model not found")

    input_id = str(uuid.uuid4())
    output_id = str(uuid.uuid4())

    input_file_path = os.path.join(section_storage_path, f"{input_id}.wav")
    output_file_path = os.path.join(section_storage_path, f"{output_id}_processed.wav")

    with open(input_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Gọi hàm infer để xử lý âm thanh
    infer(
        input_path=input_file_path,  # Sử dụng tệp đã lưu
        output_path=Path(output_file_path),
        model_path=Path(model_info.model_path),
        config_path=Path(model_info.config_path),
        speaker=0,
        cluster_model_path=Path(model_info.cluster_model_path),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
