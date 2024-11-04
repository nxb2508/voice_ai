from so_vits_svc_fork.preprocessing.preprocess_split import preprocess_split

if __name__ == "__main__":
    input_directory = "D:/DATN/Code/dataset_raw_raw"  # Thay đổi đường dẫn cho phù hợp
    output_directory = "D:/DATN/Code/dataset_raw/sontung1"  # Thư mục lưu kết quả
    sample_rate = 22050  # Tốc độ lấy mẫu, có thể thay đổi theo yêu cầu

    preprocess_split(
        input_dir=input_directory,
        output_dir=output_directory,
        sr=sample_rate,
        max_length=10.0,
        top_db=30,
        frame_seconds=0.5,
        hop_seconds=0.1,
        n_jobs=-1,  # Sử dụng tất cả các CPU có sẵn
    )
