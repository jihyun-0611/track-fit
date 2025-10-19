import cv2
import json
import numpy as np
import os
from config import DATA_DIR


COCO_SKELETON = [
    [0, 1], [0, 2],           # nose to eyes
    [1, 3], [2, 4],           # eyes to ears
    [0, 5], [0, 6],           # nose to shoulders
    [5, 7], [7, 9],           # left arm
    [6, 8], [8, 10],          # right arm
    [5, 6],                   # shoulders
    [5, 11], [6, 12],         # shoulders to hips
    [11, 12],                 # hips
    [11, 13], [13, 15],       # left leg
    [12, 14], [14, 16]        # right leg
]

# 색상 (BGR)
COLORS = {
    'keypoint': (0, 255, 0),      
    'skeleton': (255, 0, 0),     
    'bbox': (0, 0, 255),           
    'text': (255, 255, 255)       
}


def draw_keypoints_on_frame(frame, pose_data, min_confidence=0.3):
    """
    Returns:
        annotated_frame: 키포인트가 그려진 프레임
    """
    annotated_frame = frame.copy()
    
    if not pose_data:
        return annotated_frame
    
    keypoints = pose_data['keypoints']
    bbox = pose_data['bbox']
    confidence = pose_data['confidence']
    
    # Bounding Box 
    cv2.rectangle(
        annotated_frame,
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        COLORS['bbox'],
        2
    )
    
    cv2.putText(
        annotated_frame,
        f"Conf: {confidence:.2f}",
        (bbox[0], bbox[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLORS['text'],
        2
    )
    
    for connection in COCO_SKELETON:
        idx1, idx2 = connection
        kp1 = keypoints[idx1]
        kp2 = keypoints[idx2]
        
        # 신뢰도가 충분히 높은 경우만
        if kp1['confidence'] > min_confidence and kp2['confidence'] > min_confidence:
            pt1 = (int(kp1['x']), int(kp1['y']))
            pt2 = (int(kp2['x']), int(kp2['y']))
            cv2.line(annotated_frame, pt1, pt2, COLORS['skeleton'], 2)
    
    # 키포인트 
    for idx, kp in enumerate(keypoints):
        if kp['confidence'] > min_confidence:
            center = (int(kp['x']), int(kp['y']))
 
            radius = int(5 + kp['confidence'] * 3)
            cv2.circle(annotated_frame, center, radius, COLORS['keypoint'], -1)
            # 키포인트 번호 표시 
            # cv2.putText(annotated_frame, str(idx), center, 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return annotated_frame


def visualize_video_with_keypoints(video_path, json_path, output_path=None, 
                                   show_video=True, fps=30):

    with open(json_path, 'r', encoding='utf-8') as f:
        keypoints_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path}를 열 수 없습니다.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"비디오 정보:")
    print(f"  해상도: {frame_width}x{frame_height}")
    print(f"  총 프레임: {total_frames}")
    print(f"  키포인트 데이터 프레임: {len(keypoints_data)}")
    
    # VideoWriter
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                (frame_width, frame_height))
    
    frame_idx = 0
    frames_with_pose = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(keypoints_data):
            frame_data = keypoints_data[frame_idx]
            poses = frame_data.get('poses', [])
            
            if poses:
                frames_with_pose += 1
                for pose in poses:
                    frame = draw_keypoints_on_frame(frame, pose)
            
            info_text = f"Frame: {frame_idx} | Poses: {len(poses)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if show_video:
            cv2.imshow('Keypoints Visualization', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("사용자가 중단했습니다.")
                break
        
        if writer:
            writer.write(frame)
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
    if show_video:
        cv2.destroyAllWindows()
    

    detection_rate = (frames_with_pose / total_frames * 100) if total_frames > 0 else 0
    print(f"\n처리 완료:")
    print(f"  포즈 감지 프레임: {frames_with_pose}/{total_frames}")
    print(f"  감지율: {detection_rate:.1f}%")
    if output_path:
        print(f"  저장 경로: {output_path}")


def main():
    
    video_name = "bench press_57"  # 파일명 (확장자 제외)
    exercise_type = "bench press"   # 운동 타입
    
    video_path = os.path.join(DATA_DIR, "sample_videos", exercise_type, 
                             f"{video_name}.mp4")
    json_path = os.path.join(DATA_DIR, "keypoints_mediapipe_new", "json", 
                            exercise_type, f"{video_name}.json")
    output_path = os.path.join(DATA_DIR, "visualizations_new", 
                              f"{video_name}_visualized.mp4")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} 파일을 찾을 수 없습니다.")
        return
    if not os.path.exists(json_path):
        print(f"Error: {json_path} 파일을 찾을 수 없습니다.")
        return
    
    print(f"비디오 시각화 시작...")
    print(f"원본 비디오: {video_path}")
    print(f"키포인트 데이터: {json_path}")

    
    visualize_video_with_keypoints(
        video_path=video_path,
        json_path=json_path,
        output_path=output_path,  # None으로 설정하면 저장 안 함
        show_video=True,          # False로 설정하면 화면에 표시 안 함
        fps=30
    )


if __name__ == '__main__':
    main()