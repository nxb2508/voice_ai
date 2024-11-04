# from scipy.io import wavfile
# import os
# import numpy as np
# import argparse
# from tqdm import tqdm
# import json

# from datetime import datetime, timedelta

# # Utility functions

# def GetTime(video_seconds):

#     if (video_seconds < 0) :
#         return 00

#     else:
#         sec = timedelta(seconds=float(video_seconds))
#         d = datetime(1,1,1) + sec

#         instant = str(d.hour).zfill(2) + ':' + str(d.minute).zfill(2) + ':' + str(d.second).zfill(2) + str('.001')

#         return instant

# def GetTotalTime(video_seconds):

#     sec = timedelta(seconds=float(video_seconds))
#     d = datetime(1,1,1) + sec
#     delta = str(d.hour) + ':' + str(d.minute) + ":" + str(d.second)

#     return delta

# def windows(signal, window_size, step_size):
#     if type(window_size) is not int:
#         raise AttributeError("Window size must be an integer.")
#     if type(step_size) is not int:
#         raise AttributeError("Step size must be an integer.")
#     for i_start in range(0, len(signal), step_size):
#         i_end = i_start + window_size
#         if i_end >= len(signal):
#             break
#         yield signal[i_start:i_end]

# def energy(samples):
#     return np.sum(np.power(samples, 2.)) / float(len(samples))

# def rising_edges(binary_signal):
#     previous_value = 0
#     index = 0
#     for x in binary_signal:
#         if x and not previous_value:
#             yield index
#         previous_value = x
#         index += 1

# '''
# Last Acceptable Values

# min_silence_length = 0.3
# silence_threshold = 1e-3
# step_duration = 0.03/10

# '''
# # Change the arguments and the input file here
# input_file = 'D:\\DATN\\Code\\sontung.wav'
# output_dir = 'D:\\DATN\\Code\\dataset_raw\\sontung'
# min_silence_length = 0.1  # The minimum length of silence at which a split may occur [seconds]. Defaults to 3 seconds.
# silence_threshold = 1e-4  # The energy level (between 0.0 and 1.0) below which the signal is regarded as silent.
# step_duration = 0.03/10   # The amount of time to step forward in the input file after calculating energy. Smaller value = slower, but more accurate silence detection. Larger value = faster, but might miss some split opportunities. Defaults to (min-silence-length / 10.).


# input_filename = input_file
# window_duration = min_silence_length
# if step_duration is None:
#     step_duration = window_duration / 10.
# else:
#     step_duration = step_duration

# output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
# dry_run = False

# print("Splitting {} where energy is below {}% for longer than {}s.".format(
#     input_filename,
#     silence_threshold * 100.,
#     window_duration
#     )
# )

# # Read and split the file

# sample_rate, samples = input_data=wavfile.read(filename=input_filename, mmap=True)

# max_amplitude = np.iinfo(samples.dtype).max
# print(max_amplitude)

# max_energy = energy([max_amplitude])
# print(max_energy)

# window_size = int(window_duration * sample_rate)
# step_size = int(step_duration * sample_rate)

# signal_windows = windows(
#     signal=samples,
#     window_size=window_size,
#     step_size=step_size
# )

# window_energy = (energy(w) / max_energy for w in tqdm(
#     signal_windows,
#     total=int(len(samples) / float(step_size))
# ))

# window_silence = (e > silence_threshold for e in window_energy)

# cut_times = (r * step_duration for r in rising_edges(window_silence))

# # This is the step that takes long, since we force the generators to run.
# print("Finding silences...")
# cut_samples = [int(t * sample_rate) for t in cut_times]
# cut_samples.append(-1)

# cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]

# video_sub = {str(i) : [str(GetTime(((cut_samples[i])/sample_rate))),
#                        str(GetTime(((cut_samples[i+1])/sample_rate)))]
#              for i in range(len(cut_samples) - 1)}

# for i, start, stop in tqdm(cut_ranges):
#     output_file_path = "{}_{:03d}.wav".format(
#         os.path.join(output_dir, output_filename_prefix),
#         i
#     )
#     if not dry_run:
#         print("Writing file {}".format(output_file_path))
#         wavfile.write(
#             filename=output_file_path,
#             rate=sample_rate,
#             data=samples[start:stop]
#         )
#     else:
#         print("Not writing file {}".format(output_file_path))

# with open (output_dir+'\\'+output_filename_prefix+'.json', 'w') as output:
#     json.dump(video_sub, output)
import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import json
from datetime import datetime, timedelta

# Các thông số cấu hình
input_file = "D:\\DATN\\Code\\vocals.wav"  # Đường dẫn tệp âm thanh đầu vào
output_dir = "D:\\DATN\\Code\\dataset_raw\\sontung"  # Thư mục đầu ra
min_silence_length = 0.1  # Độ dài tối thiểu của khoảng lặng (giây)
silence_threshold = 1e-4  # Ngưỡng năng lượng để xác định khoảng lặng
step_duration = 0.03 / 10  # Bước nhảy khi tính năng lượng (giây)
min_duration = 5  # Độ dài tối thiểu của mỗi đoạn cắt (giây)
max_duration = 10  # Độ dài tối đa của mỗi đoạn cắt (giây)
start_index = 26

# Hàm hỗ trợ tính thời gian
def get_time(seconds):
    sec = timedelta(seconds=float(seconds))
    d = datetime(1, 1, 1) + sec
    return (
        f"{str(d.hour).zfill(2)}:{str(d.minute).zfill(2)}:{str(d.second).zfill(2)}.001"
    )


# Hàm cắt tín hiệu thành các cửa sổ nhỏ
def windows(signal, window_size, step_size):
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]


# Hàm tính năng lượng của tín hiệu
def energy(samples):
    return np.sum(np.power(samples, 2.0)) / float(len(samples))


# Hàm tìm các điểm bắt đầu của khoảng lặng
def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1


# Đọc tệp âm thanh
sample_rate, samples = wavfile.read(filename=input_file, mmap=True)
max_amplitude = np.iinfo(samples.dtype).max
max_energy = energy([max_amplitude])

# Thiết lập các tham số cửa sổ và bước nhảy
window_size = int(min_silence_length * sample_rate)
step_size = int(step_duration * sample_rate)

# Tính năng lượng của các cửa sổ
signal_windows = windows(signal=samples, window_size=window_size, step_size=step_size)
window_energy = (
    energy(w) / max_energy
    for w in tqdm(signal_windows, total=int(len(samples) / float(step_size)))
)
window_silence = (e > silence_threshold for e in window_energy)

# Xác định các khoảng cắt
cut_times = (r * step_duration for r in rising_edges(window_silence))
cut_samples = [int(t * sample_rate) for t in cut_times]
cut_samples.append(-1)  # Đánh dấu kết thúc

# Tạo các đoạn cắt với độ dài từ 5 đến 10 giây
cut_ranges = []
for i in range(len(cut_samples) - 1):
    start = cut_samples[i]
    stop = cut_samples[i + 1]
    segment_duration = (stop - start) / sample_rate

    # Nếu đoạn quá dài, chia nhỏ
    if segment_duration > max_duration:
        num_subsegments = int(segment_duration // max_duration) + 1
        subsegment_size = int((stop - start) / num_subsegments)

        for j in range(num_subsegments):
            sub_start = start + j * subsegment_size
            sub_stop = sub_start + subsegment_size
            cut_ranges.append((len(cut_ranges), sub_start, sub_stop))
    elif segment_duration >= min_duration:
        cut_ranges.append((len(cut_ranges), start, stop))

# Tạo cấu trúc dữ liệu lưu thời gian bắt đầu và kết thúc của mỗi đoạn
video_sub = {
    str(i): [
        get_time(cut_ranges[i][1] / sample_rate),
        get_time(cut_ranges[i][2] / sample_rate),
    ]
    for i in range(len(cut_ranges))
}

# Xuất các đoạn cắt thành tệp âm thanh
os.makedirs(output_dir, exist_ok=True)
output_filename_prefix = os.path.splitext(os.path.basename(input_file))[0]
for i, start, stop in tqdm(cut_ranges):
    output_file_path = f"{os.path.join(output_dir, output_filename_prefix)}_{i+start_index:03d}.wav"
    print(f"Writing file {output_file_path}")
    wavfile.write(filename=output_file_path, rate=sample_rate, data=samples[start:stop])

# Lưu file JSON với thời gian bắt đầu và kết thúc của mỗi đoạn
with open(os.path.join(output_dir, f"{output_filename_prefix}.json"), "w") as output:
    json.dump(video_sub, output)
