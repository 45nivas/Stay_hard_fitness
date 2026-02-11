"""
Camera Optimizer for Zero-Buffering Video Streaming
Eliminates camera buffering issues with aggressive optimization techniques
"""

import cv2
import time
import threading
from queue import Queue, Empty
import numpy as np


class ZeroBufferCamera:
    """
    High-performance camera class that eliminates buffering
    Uses background thread to continuously grab fresh frames
    """
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.frame_queue = Queue(maxsize=2)  # Very small queue
        self.running = False
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance metrics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.start_time = time.time()
    
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        print("🎥 Initializing zero-buffer camera...")
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # CRITICAL: Force minimal buffer size
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Optimized resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        # Lower FPS for stability
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Use MJPG for hardware acceleration
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Disable auto-exposure for consistent timing
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # Additional optimizations
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        
        # Aggressive warm-up sequence
        print("🔥 Warming up camera (this may take a moment)...")
        for i in range(15):
            ret, frame = self.cap.read()
            if ret:
                print(f"   Frame {i+1}/15 captured")
            else:
                print(f"   Frame {i+1}/15 failed")
        
        print("✅ Camera initialized successfully!")
        return True
    
    def _capture_worker(self):
        """Background thread that continuously grabs fresh frames"""
        frame_count = 0
        last_fps_time = time.time()
        
        while self.running:
            try:
                # Always grab to clear buffer
                self.cap.grab()
                
                # Retrieve the actual frame
                ret, frame = self.cap.retrieve()
                
                if ret and frame is not None:
                    # Update latest frame with thread safety
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.frames_captured += 1
                    
                    # Drop old frames from queue
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                            self.frames_dropped += 1
                        except Empty:
                            break
                    
                    # Add new frame
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        self.frames_dropped += 1
                
                frame_count += 1
                
                # Print performance stats every 5 seconds
                if frame_count % 75 == 0:  # ~5 seconds at 15 FPS
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    actual_fps = 75 / elapsed if elapsed > 0 else 0
                    print(f"📊 Camera FPS: {actual_fps:.1f} | Dropped: {self.frames_dropped} | Total: {self.frames_captured}")
                    last_fps_time = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Camera capture error: {e}")
                continue
    
    def start(self):
        """Start the camera with background capture"""
        if not self.initialize_camera():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        
        # Wait for first frame
        timeout = 5  # 5 second timeout
        start_wait = time.time()
        while self.latest_frame is None and (time.time() - start_wait) < timeout:
            time.sleep(0.1)
        
        if self.latest_frame is None:
            print("❌ Failed to capture first frame within timeout")
            return False
        
        print("🚀 Zero-buffer camera started successfully!")
        return True
    
    def get_frame(self):
        """Get the latest fresh frame (zero buffer)"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        
        return False, None
    
    def get_performance_stats(self):
        """Get camera performance statistics"""
        elapsed = time.time() - self.start_time
        capture_fps = self.frames_captured / elapsed if elapsed > 0 else 0
        drop_rate = (self.frames_dropped / max(1, self.frames_captured)) * 100
        
        return {
            'fps': capture_fps,
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'drop_rate': drop_rate,
            'uptime': elapsed
        }
    
    def stop(self):
        """Stop camera and cleanup"""
        print("🛑 Stopping zero-buffer camera...")
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        # Print final stats
        stats = self.get_performance_stats()
        print(f"📈 Final stats - FPS: {stats['fps']:.1f}, Drop rate: {stats['drop_rate']:.1f}%")
        print("✅ Camera stopped successfully")
    
    def __enter__(self):
        """Context manager entry"""
        if self.start():
            return self
        else:
            raise RuntimeError("Failed to start camera")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def create_optimized_camera(camera_index=0):
    """
    Factory function to create an optimized camera instance
    
    Args:
        camera_index: Camera index (usually 0 for default camera)
    
    Returns:
        ZeroBufferCamera instance or None if failed
    """
    try:
        camera = ZeroBufferCamera(camera_index)
        if camera.start():
            return camera
        else:
            return None
    except Exception as e:
        print(f"Failed to create optimized camera: {e}")
        return None


# Testing function
def test_camera_performance():
    """Test the camera performance"""
    print("🧪 Testing zero-buffer camera performance...")
    
    try:
        with ZeroBufferCamera() as camera:
            print("Camera started, testing for 10 seconds...")
            
            start_test = time.time()
            frame_count = 0
            
            while (time.time() - start_test) < 10:
                ret, frame = camera.get_frame()
                if ret:
                    frame_count += 1
                    
                    # Show frame every 30 frames
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count} - Shape: {frame.shape}")
                
                time.sleep(0.05)  # 20 FPS test
            
            # Final performance report
            stats = camera.get_performance_stats()
            print(f"\n🎯 Test Results:")
            print(f"   • Capture FPS: {stats['fps']:.1f}")
            print(f"   • Drop Rate: {stats['drop_rate']:.1f}%")
            print(f"   • Frames Processed: {frame_count}")
            print(f"   • Test Duration: {stats['uptime']:.1f}s")
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")


if __name__ == "__main__":
    test_camera_performance()