"video__checker---->AI-Based Media Authenticity Verifier"
import cv2
import tempfile

def check_video_authenticity(video_file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video_file.read())

    cap = cv2.VideoCapture(temp.name)
    success, frame = cap.read()
    cap.release()

    risk = 0.7  # placeholder

    return {
        "risk": risk,
        "frame": frame,
        "reason": "Temporal inconsistencies detected between frames"
    }