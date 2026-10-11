import cv2, numpy as np
import matplotlib.pyplot as plt

def analyze_motion(path,output_plot,sample_every=5):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Unable to open video.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx=0; prev=None; times=[]; motion=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%sample_every:
            idx+=1; continue
        gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(320,180))
        if prev is not None:
            flow=cv2.calcOpticalFlowFarneback(prev,gray,None,0.5,3,15,3,5,1.2,0)
            mag=np.sqrt(flow[...,0]**2+flow[...,1]**2)
            times.append(idx/fps); motion.append(float(np.mean(mag)))
        prev=gray; idx+=1
    cap.release()
    plt.figure(figsize=(10,4)); plt.plot(times,motion)
    plt.xlabel("Time (seconds)"); plt.ylabel("Average optical-flow magnitude")
    plt.title("Video Motion Analysis"); plt.tight_layout(); plt.savefig(output_plot,dpi=150); plt.close()
    return {"average_motion":float(np.mean(motion)) if motion else 0.0}
