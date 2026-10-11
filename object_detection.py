import os, cv2
from collections import Counter

def detect_objects(path,output_image,model_name="yolo11n.pt"):
    from ultralytics import YOLO
    model=YOLO(model_name)
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Unable to open video.")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n=min(12,max(1,total))
    indices=[int(i*max(total-1,0)/max(n-1,1)) for i in range(n)]
    counts=Counter(); last=None
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
        if not ok: continue
        result=model.predict(frame,verbose=False)[0]
        if result.boxes is not None:
            for cls in result.boxes.cls.tolist(): counts[result.names[int(cls)]]+=1
        last=result.plot()
    cap.release()
    if last is None: raise ValueError("No frame processed.")
    os.makedirs(os.path.dirname(output_image),exist_ok=True); cv2.imwrite(output_image,last)
    return dict(counts)
