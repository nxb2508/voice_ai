# Hướng dẫn Cài đặt

- Sử dụng python 3.10.15

## Bước 1: Tạo môi trường ảo
```bash
python3.11 -m venv venv
```
- Kích hoạt môi trường ảo
```bash
source venv/bin/activate
```
## Bước 2: Cài đặt thư viện

- cài thư viện pytorch và so-vits-svc

```bash
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U so-vits-svc-fork
```

- cài theo file requirements.txt

```bash
pip install -r requirements.txt
```