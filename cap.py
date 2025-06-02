import cv2
import os
import time

base_dir = "./data/images"
frame_width = 640
frame_height = 640
fps = 30.0
total_images = 120
interval = 0.5  # 저장 간격 (초 단위)

i = 1
while os.path.exists(os.path.join(base_dir, f"try{i}")):
    i += 1
try_dir = os.path.join(base_dir, f"try{i}")
frames_dir = os.path.join(try_dir, "frames")
os.makedirs(frames_dir, exist_ok=False)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(frames_dir, fourcc, fps, (frame_width, frame_height))

# 카메라 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 이미지 캡처 루프
for img_count in range(total_images):
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 이미지 저장
    filename = os.path.join(frames_dir, f"img_{img_count:04}.jpg")
    cv2.imwrite(filename, frame)
    print(f"{filename} 저장 완료")

    out.write(frame)

    # 실시간 웹캠 화면 출력
    cv2.imshow("Webcam Live", frame)

    # 키보드 입력 체크 (ESC = 27)
    if cv2.waitKey(1) & 0xFF == 27:
        print("사용자 ESC로 중단")
        break

    time.sleep(interval)

print("✅ 1000장 캡처 완료 또는 중단됨")

# 자원 정리
cap.release()
cv2.destroyAllWindows()