# Hướng dẫn Cài đặt

## Bước 1: Cài đặt thư viện

- cài theo fiel requirements.txt
- cài xong thì mở cmd chạy 'svcg' nếu hiện ra giao diện ![alt text](assets/images/image.png) là thành công

## Bước 2: Tải dataset

- vào link : https://drive.google.com/drive/folders/1rb2syP4U7jYz_z9WoG7MVR6t3i375smZ?usp=sharing
- tải folder dataset về đổi tên thành dataset_raw

## Bước 3

- Tìm kiếm folder thư viện so-vits-svc-fork mà mình cài ở máy ảo
- cho folder dataset_raw vào trong đó như hình ![alt text](assets/images/image-1.png)
- mở cmd tại đó luôn
- chạy lần lượt 4 lệnh
  - svc pre-resample
  - svc pre-config
  - svc pre-hubert
  - svc train -t

# Hướng dẫn cắt audio

- ![alt text](assets/images/image-4.png) sửa 2 cái này. audio cần cắt cứ nhét hết vào file dataset_raw_raw.
- Folder output thì trong dataset_raw hãy tạo 1 folder trên ca sĩ rồi cho output đã split vô đó


# Endpoints api
- GET /models?category=?
  - Input: 
  - OutPut:
    - id_model: int
    - model_name: str
    - model_path: str
    - config_path: str
    - cluster_model_path: str
    - category :str
- POST /text-to-speech/
  - Input: text: str
  - OutPut: output.wav 
- POST /text-file-to-speech/
  - Input: file .docx/.txt
  - OutPut: output.wav
- POST /text-file-to-speech-and-infer/
  - Input : 
    - file .docx/.txt
    - model_id: int
  - Output: output.wav
- POST /text-to-speech-and-infer/
  - Input: 
    - text: str
    - model_id: int
  - Output: output.wav
- POST /infer-audio/
  - Input: 
    - file audio.wav
    - model_id: int
  - Ouput: output.wav
- POST /train-model/
  - Input: 
    - file audio.wav
    - name: str
    - f0_method: ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] (chọn 1)
  - OutPut: Tạm thời là "message": "Audio processing completed successfully!"