"""
=============================================================================
MODULE NAME: sensor.py
PROJECT: DriveMind AI // Spatial Telemetry & Congestion Engine
DESCRIPTION: 
    Strict spatial sensor that detects vehicles and pedestrians across zones.
    If ANY vehicle or object is detected on the left or right, it is immediately 
    marked as blocked (not clear) to prevent dangerous false maneuver advice.
=============================================================================
"""

import numpy as np

VEHICLE_CLASS_LABELS = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle (Bike)",
    5: "Bus",
    7: "Truck",
}


def get_detection_label(class_id):
    if class_id == 0:
        return "Person"
    return VEHICLE_CLASS_LABELS.get(class_id)


class VisionSensor:
    def __init__(self):
        """
        Vision Sensor module for precise spatial mapping and zone partitioning.
        """
        pass

    def scan_environment(self, boxes, classes, frame_width, frame_height):
        """
        Scans all detected bounding boxes from YOLO, partitions them into Center, 
        Left, and Right zones, and strictly flags blocked areas.
        """
        sensor_report = {
            "center_vehicle": False,
            "is_too_close": False,
            "left_blocked": False,
            "right_blocked": False,
            "left_person": False,
            "right_person": False,
            "center_person": False,
            "detected_objects": [],
            "total_objects_count": 0
        }

        center_max_h = 0
        left_cluttered = False
        right_cluttered = False

        if boxes is not None and len(boxes) > 0:
            sensor_report["total_objects_count"] = len(boxes)
            
            for box, cls in zip(boxes, classes):
                cls_id = int(cls)
                x1, y1, x2, y2 = box
                box_height = y2 - y1
                x_center = (x1 + x2) / 2

                # Precise Zone Partitioning
                is_center = (frame_width * 0.35) < x_center < (frame_width * 0.65)
                is_left = x_center <= (frame_width * 0.35)
                is_right = x_center >= (frame_width * 0.35)

                label = get_detection_label(cls_id)
                if label == "Person":
                    sensor_report["detected_objects"].append(label)
                    if is_center:
                        sensor_report["center_person"] = True
                    elif is_left:
                        sensor_report["left_person"] = True
                        left_cluttered = True
                    elif is_right:
                        sensor_report["right_person"] = True
                        right_cluttered = True

                elif label is not None:
                    sensor_report["detected_objects"].append(label)

                    if is_center:
                        sensor_report["center_vehicle"] = True
                        if box_height > center_max_h:
                            center_max_h = box_height
                    elif is_left:
                        left_cluttered = True
                    elif is_right:
                        right_cluttered = True

            # STRICT RULE: Agar left ya right mein ek bhi object/vehicle hai toh use blocked mark karo!
            sensor_report["left_blocked"] = left_cluttered
            sensor_report["right_blocked"] = right_cluttered

            # Proximity check for center obstacle
            if center_max_h > (frame_height * 0.30):
                sensor_report["is_too_close"] = True

        return sensor_report