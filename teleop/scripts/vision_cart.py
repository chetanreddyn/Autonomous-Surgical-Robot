import cv2

class VideoStream:
    def __init__(self, device_numbers):
        self.device_numbers = device_numbers
        self.caps = self.create_caps(device_numbers)

    def create_caps(self, device_numbers):
        caps = []
        for device_number in device_numbers:
            gst_pipeline = f"decklinkvideosrc device-number={device_number} ! videoconvert ! appsink"
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise ValueError(f"Error: Unable to open video stream for device {device_number}")
            caps.append(cap)
        return caps

    def display_stream(self, stream_index):
        cap = self.caps[stream_index]
        print(f"Displaying stream from device {self.device_numbers[stream_index]}. Press 'q' to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to capture frame from device {self.device_numbers[stream_index]}")
                break
            cv2.imshow(f"Frame {self.device_numbers[stream_index]}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_all_streams(self):
        print("Displaying all streams. Press 'q' to exit.")
        while True:
            for i, cap in enumerate(self.caps):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Failed to capture frame from device {self.device_numbers[i]}")
                    return
                cv2.imshow(f"Frame {i}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    device_numbers = {0: "Right Camera", 1: "Left Camera"}  # List of device numbers

    video_stream = VideoStream(device_numbers)
    
    # Display streams individually
    # video_stream.display_stream(0)  # Display stream from device 0
    # video_stream.display_stream(1)  # Display stream from device 1
    
    # Alternatively, display all streams together
    video_stream.display_all_streams()