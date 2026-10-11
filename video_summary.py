import os, cv2, numpy as np

def find_important_frames(path,output_dir,max_frames=12,sample_every=10):
    os.makedirs(output_dir,exist_ok=True)
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Unable to open video.")
    candidates=[]; prev=None; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%sample_every:
            idx+=1; continue
        gray=cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2GRAY)
        if prev is not None: candidates.append((float(np.mean(cv2.absdiff(prev,gray))),idx,frame.copy()))
        prev=gray; idx+=1
    cap.release()
    candidates.sort(reverse=True,key=lambda x:x[0])
    chosen=[]; gap=max(1,idx//max(max_frames,1)//2)
    for score,frame_idx,frame in candidates:
        if all(abs(frame_idx-old)>=gap for old,_ in chosen): chosen.append((frame_idx,frame))
        if len(chosen)>=max_frames: break
    chosen.sort(key=lambda x:x[0])
    paths=[]
    for i,(frame_idx,frame) in enumerate(chosen,1):
        p=os.path.join(output_dir,f"important_{i:02d}_frame_{frame_idx}.jpg")
        cv2.imwrite(p,frame); paths.append(p)
    return chosen,paths
