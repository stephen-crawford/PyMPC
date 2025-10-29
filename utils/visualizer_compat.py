class ROSLine:
    def __init__(self, name: str, frame_id: str = "map"):
        self.name = name
        self.frame_id = frame_id

    def publish(self, points, color=(0.0, 0.0, 1.0), width=0.02):
        # No-op in test environment
        return

    def clear(self):
        # No-op in test environment
        return

class ROSPointMarker:
    def __init__(self, name: str, frame_id: str = "map"):
        self.name = name
        self.frame_id = frame_id

    def publish(self, point, color=(1.0, 0.0, 0.0), size=0.05):
        # No-op in test environment
        return


