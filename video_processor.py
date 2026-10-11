import os, cv2

def video_info(path):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps":fps,"frames":frames,"width":width,"height":height,"duration":frames/fps}

def make_summary_video(source, output, selected_frames, seconds_per_frame=0.8):
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise ValueError("Unable to open source video.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer=cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    repeat=max(1,int(fps*seconds_per_frame))
    for frame in selected_frames:
        if frame is not None:
            frame=cv2.resize(frame,(w,h))
            for _ in range(repeat): writer.write(frame)
    writer.release()
    return output
