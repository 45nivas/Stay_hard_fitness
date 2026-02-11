"""
EMERGENCY Raw Camera Mode - Bypasses all processing for maximum speed
Use this when all other optimizations fail
"""

import cv2
import time
from django.http import StreamingHttpResponse

def raw_camera_stream():
    """
    ABSOLUTE MINIMAL camera processing - no pose detection, no rep counting
    Only raw camera feed with lowest possible latency
    """
    print("🚨 LAUNCHING RAW CAMERA MODE - MAXIMUM SPEED")
    
    # Try all Windows backends
    backends = [
        cv2.CAP_DSHOW,    # DirectShow (usually fastest on Windows)
        cv2.CAP_MSMF,     # Media Foundation 
        cv2.CAP_V4L2,     # Linux fallback
        cv2.CAP_ANY       # System default
    ]
    
    cap = None
    for i, backend in enumerate(backends):
        try:
            print(f"🔄 Trying backend {i+1}/4...")
            cap = cv2.VideoCapture(0, backend)
            if cap and cap.isOpened():
                print(f"✅ Backend {i+1} success!")
                break
            if cap:
                cap.release()
        except Exception as e:
            print(f"❌ Backend {i+1} failed: {e}")
            continue
    
    if not cap or not cap.isOpened():
        print("💥 ALL BACKENDS FAILED!")
        return
    
    try:
        # ABSOLUTE MINIMUM settings for speed
        print("⚡ Setting EXTREME low-latency mode...")
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Single frame buffer
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)   # Tiny resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)  # Tiny resolution
        cap.set(cv2.CAP_PROP_FPS, 5)             # Very low FPS
        
        # Disable ALL automatic features
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        print("🔥 Extreme camera flush...")
        # Flush old frames
        for i in range(30):
            cap.grab()
            if i % 10 == 0:
                print(f"   Flushing: {i+1}/30")
        
        print("🚀 RAW MODE ACTIVE")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Grab latest frame only
            if not cap.grab():
                continue
                
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            
            frame_count += 1
            
            # ZERO PROCESSING - just flip and send
            frame = cv2.flip(frame, 1)
            
            # Add minimal status
            cv2.putText(frame, f'RAW:{frame_count}', (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Lowest quality encoding for speed
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 20])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Stats every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 RAW MODE: {fps:.1f} FPS, {frame_count} frames")
            
            # Minimal delay
            time.sleep(0.02)
            
    except Exception as e:
        print(f"💥 Raw camera error: {e}")
    finally:
        if cap:
            cap.release()
        print("🛑 Raw camera mode stopped")


def emergency_video_feed(request, workout_name):
    """Emergency endpoint for raw camera feed"""
    return StreamingHttpResponse(
        raw_camera_stream(), 
        content_type='multipart/x-mixed-replace; boundary=frame'
    )