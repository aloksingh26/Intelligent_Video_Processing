import cv2

def _hist(frame):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
    cv2.normalize(h,h)
    return h

def detect_scenes(path,sample_every=10,threshold=0.45):
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Unable to open video.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    scenes=[]; prev=None; idx=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx%sample_every:
            idx+=1; continue
        cur=_hist(frame)
        if prev is not None:
            d=cv2.compareHist(prev,cur,cv2.HISTCMP_BHATTACHARYYA)
            if d>=threshold: scenes.append({"frame":idx,"time":round(idx/fps,2),"score":round(float(d),4)})
        prev=cur; idx+=1
    cap.release()
    return scenes
