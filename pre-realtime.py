from ultralytics import YOLO
import cv2
import os
import time
import numpy as np
from collections import defaultdict

realtime = True  # False: 비디오 사용, True: 카메라 사용

# 기본 설정
model_path = "best.pt"
tracker_path = "bytetrack.yaml"
sample_video_path = "sample_video2.mp4"
resize_shape = (640, 640)
conf_threshold = 0.35
duration_limit = 60  # 최대 실행 시간 (초)

# 출력 디렉토리 생성
base_dir = "./prediction_output"
os.makedirs(base_dir, exist_ok=True)

i = 1
while os.path.exists(os.path.join(base_dir, f"try{i}")):
    i += 1
try_dir = os.path.join(base_dir, f"try{i}")
os.makedirs(try_dir, exist_ok=True)

pred_frames_dir = os.path.join(try_dir, "frames")
os.makedirs(pred_frames_dir, exist_ok=False)

original_frames_dir = os.path.join(try_dir, "original_frames")
os.makedirs(original_frames_dir, exist_ok=False)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
original_video_path = os.path.join(try_dir, "original.mp4")
original_video_writer = cv2.VideoWriter(original_video_path, fourcc, 30.0, resize_shape)

pred_video_path = os.path.join(try_dir, "prediction.mp4")
pred_video_writer = cv2.VideoWriter(pred_video_path, fourcc, 30.0, resize_shape)

coords_txt_path = os.path.join(try_dir, "coordinates.txt")
info_txt_path = os.path.join(try_dir, "info.txt")

# 정보 저장
with open(info_txt_path, 'w') as f:
    f.write(f"model_path: {model_path}\n")
    f.write(f"resize_shape: {resize_shape}\n")
    f.write(f"conf_threshold: {conf_threshold}\n")
    f.write(f"duration_limit: {duration_limit}\n")
    f.write(f"realtime: {realtime}\n")
    f.write(f"sample_video: {sample_video_path}\n")

# 좌표 텍스트 초기화
with open(coords_txt_path, 'w') as f:
    f.write("frame,id,x_center,y_center,confidence\n")

# 모델 및 비디오 로드
model = YOLO(model_path)

if realtime == False:
    cap = cv2.VideoCapture(sample_video_path)
else:
    cap = cv2.VideoCapture(0)

# 트래킹 궤적 저장용
track_history = defaultdict(list)

# 예측 루프
frame_count = 0
start_time = time.time()

while cap.isOpened():
    if time.time() - start_time > duration_limit:
        print(f"{duration_limit}초 경과로 중단")
        break

    ret, frame = cap.read()
    if not ret:
        print("영상 끝 또는 오류")
        break

    print(f"[프레임 {frame_count}] 예측 중...")

    resized_frame = cv2.resize(frame, resize_shape)
    h, w, _ = resized_frame.shape
    cx, cy = w // 2, h // 2

    # 중앙선
    cv2.line(resized_frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
    cv2.line(resized_frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

    # YOLO 트래킹
    results = model.track(
        source=resized_frame,
        conf=conf_threshold,
        persist=True,
        stream=True,
        tracker=tracker_path
    )

    for r in results:
        # 기존처럼 박스와 라벨 그려진 프레임 생성
        pred_frame = r.plot()

        # 중앙선 다시 그리기
        cv2.line(pred_frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
        cv2.line(pred_frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)

        # 트래킹된 객체 처리
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            track_id = int(box.id[0].item()) if box.id is not None else -1
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)

            # 좌표 기록
            with open(coords_txt_path, 'a') as f:
                f.write(f"{frame_count},{track_id},{x_center},{y_center},{conf:.3f}\n")

            # 이동 궤적 기록
            track_history[track_id].append((x_center, y_center))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            # 궤적 선 시각화
            pts = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(pred_frame, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

        # 프레임 저장
        original_resized = cv2.resize(frame, resize_shape)
        cv2.imwrite(os.path.join(original_frames_dir, f"frame_{frame_count:04d}.png"), original_resized)
        original_video_writer.write(original_resized)
        cv2.imwrite(os.path.join(pred_frames_dir, f"frame_{frame_count:04d}.png"), pred_frame)
        pred_video_writer.write(pred_frame)

        # cv2.imshow("original", original_resized)
        cv2.imshow("predictions", pred_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("사용자 ESC로 중단")
            break

    frame_count += 1
    time.sleep(0.05)

# 자원 정리
cap.release()
pred_video_writer.release()
original_video_writer.release()
cv2.destroyAllWindows()

# 완료 메시지
print(f"\n✅ try{i} 저장 완료")
print(f"🕒 전체 실행 시간: {time.time() - start_time:.2f}초")