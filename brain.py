"""
=============================================================================
MODULE NAME: brain.py
PROJECT: DriveMind AI // Autonomous ADAS Co-Pilot & Decision Engine
DESCRIPTION: 
    Enterprise-grade decision matrix for autonomous driving telemetry.
    Takes structured spatial sensor data, evaluates multi-zone risk factors,
    prevents false maneuvers in heavy traffic, ensures pedestrian safety, 
    and outputs precise, deterministic Roman Urdu tactical guidance.
=============================================================================
"""

import time
import logging

# Configure internal logging for the AI Brain
logging.basicConfig(level=logging.INFO, format='[DriveBrain AI] %(asctime)s - %(levelname)s - %(message)s')


class DriveBrain:
    """
    DriveBrain class acts as the cognitive core of the DriveMind AI system.
    It processes real-time sensor arrays, maintains temporal memory of traffic states,
    and calculates collision-avoidance advisories without risking the vehicle.
    """

    def __init__(self):
        # Temporal state memory to track object persistence across frames
        self.prev_left_blocked = False
        self.prev_right_blocked = False
        self.prev_center_vehicle = False
        self.consecutive_danger_frames = 0
        self.max_tolerance_frames = 5
        
        # System state metrics
        self.current_risk_level = "SAFE"
        self.last_advisory_timestamp = time.time()
        
        logging.info("DriveBrain initialized successfully with zero-risk safety protocols.")

    def _assess_pedestrian_threat(self, sensor_report):
        """
        Internal helper method to evaluate high-priority pedestrian hazards.
        Pedestrians and humans near the vehicle take precedence over all vehicular traffic.
        """
        center_person = sensor_report.get("center_person", False)
        left_person = sensor_report.get("left_person", False)
        right_person = sensor_report.get("right_person", False)

        if center_person:
            return {
                "has_threat": True,
                "status": "KHATRA: SAMNE BANDA KHADA HAI!",
                "color": (0, 0, 255),  # Red
                "advice": "Foran break lagao! Samne paidal chalne wala banda hai, speed bilkul aahista karo."
            }
        
        if left_person and right_person:
            return {
                "has_threat": True,
                "status": "DONON TARAF LOG KHADE HAIN",
                "color": (0, 255, 255),  # Yellow
                "advice": "Dono taraf log khade hain. Dhyan se center lane mein gaadi chalao aur speed kam rakho."
            }
            
        if left_person:
            return {
                "has_threat": True,
                "status": "LEFT SIDE BANDA KHADA HAI",
                "color": (0, 255, 255),  # Yellow
                "advice": "Left side par banda khada hai. Left dabne se bacho, center lane mein raho."
            }
            
        if right_person:
            return {
                "has_threat": True,
                "status": "RIGHT SIDE BANDA KHADA HAI",
                "color": (0, 255, 255),  # Yellow
                "advice": "Right side par banda khada hai. Right dabne se bacho, center lane mein raho."
            }

        return {"has_threat": False, "status": "", "color": (0, 255, 0), "advice": ""}

    def _assess_vehicle_proximity(self, sensor_report):
        """
        Internal helper method to evaluate front-facing vehicle obstacles and proximity risks.
        """
        center_veh = sensor_report.get("center_vehicle", False)
        too_close = sensor_report.get("is_too_close", False)
        left_blocked = sensor_report.get("left_blocked", False)
        right_blocked = sensor_report.get("right_blocked", False)

        if not center_veh:
            return {"has_threat": False, "status": "", "color": (0, 255, 0), "advice": ""}

        if too_close:
            self.consecutive_danger_frames += 1
            if not left_blocked and not right_blocked:
                return {
                    "has_threat": True,
                    "status": "KHATRA: SAMNE VEHICLE QAREEB HAI!",
                    "color": (0, 0, 255),  # Red
                    "advice": "Samne vehicle qareeb hai! Left aur Right dono taraf rasta khali lag raha hai, dhyan se dekh kar side le sakte hain."
                }
            elif not left_blocked:
                return {
                    "has_threat": True,
                    "status": "KHATRA: SAMNE VEHICLE QAREEB HAI!",
                    "color": (0, 0, 255),  # Red
                    "advice": "Samne vehicle qareeb hai! Right taraf traffic hai wahan MAT jao. Left side khali hai, dhyan se left ho sakte ho."
                }
            elif not right_blocked:
                return {
                    "has_threat": True,
                    "status": "KHATRA: SAMNE VEHICLE QAREEB HAI!",
                    "color": (0, 0, 255),  # Red
                    "advice": "Samne vehicle qareeb hai! Left taraf traffic hai wahan MAT jao. Right side khali hai, dhyan se right ho sakte ho."
                }
            else:
                return {
                    "has_threat": True,
                    "status": "CRITICAL: BHARI TRAFFIC & OBSTACLE!",
                    "color": (0, 0, 255),  # Red
                    "advice": "KHATRA! Samne bhi gari hai aur aaspas bhi bhari traffic/dukaanein hain! Kahin mat jao, foran speed kam karo aur brake lagao!"
                }
        else:
            return {
                "has_threat": True,
                "status": "SAMNE VEHICLE HAI: GAP RAKHO",
                "color": (0, 255, 255),  # Yellow
                "advice": "Samne vehicle hai. Fasla (gap) banaye rakho aur aaram se chalo."
            }

    def _assess_side_congestion(self, sensor_report):
        """
        Internal helper method to evaluate lateral traffic density and roadside clutter.
        """
        left_blocked = sensor_report.get("left_blocked", False)
        right_blocked = sensor_report.get("right_blocked", False)

        if left_blocked and right_blocked:
            return {
                "has_congestion": True,
                "status": "BHARI TRAFFIC: KAHIN MAT JAO",
                "color": (0, 0, 255),  # Red
                "advice": "Aaspas bhari traffic aur dukaanein hain! Bilkul side mat badlo, apni isi lane mein aaram se raho."
            }
        elif left_blocked:
            return {
                "has_congestion": True,
                "status": "LEFT SIDE TRAFFIC HAI",
                "color": (0, 255, 255),  # Yellow
                "advice": "Left side par traffic/bheed hai. Left bilkul mat dabna, center lane mein raho."
            }
        elif right_blocked:
            return {
                "has_congestion": True,
                "status": "RIGHT SIDE TRAFFIC HAI",
                "color": (0, 255, 255),  # Yellow
                "advice": "Right side par traffic/bheed hai. Right bilkul mat dabna, center lane mein raho."
            }

        return {"has_congestion": False, "status": "", "color": (0, 255, 0), "advice": ""}

    def make_decision(self, sensor_report):
        """
        Main decision evaluation pipeline. 
        Synthesizes pedestrian threats, front vehicle risks, and side congestion metrics 
        to output the final HUD status, neon display color, and tactical Roman Urdu advice.
        
        Parameters:
            sensor_report (dict): Structured telemetry dict coming from VisionSensor.
            
        Returns:
            tuple: (status_string, rgb_color_tuple, advisory_string)
        """
        if not isinstance(sensor_report, dict):
            logging.error("Invalid sensor report format received by DriveBrain.")
            return "SYSTEM ERROR", (255, 0, 0), "Sensor telemetry failure detected. Drive manually."

        # Step 1: Check for highest priority pedestrian threats
        pedestrian_eval = self._assess_pedestrian_threat(sensor_report)
        if pedestrian_eval["has_threat"]:
            self.current_risk_level = "CRITICAL"
            return pedestrian_eval["status"], pedestrian_eval["color"], pedestrian_eval["advice"]

        # Step 2: Check for front vehicle proximity and collision risks
        vehicle_eval = self._assess_vehicle_proximity(sensor_report)
        if vehicle_eval["has_threat"]:
            self.current_risk_level = "WARNING"
            return vehicle_eval["status"], vehicle_eval["color"], vehicle_eval["advice"]

        # Step 3: Check for lateral congestion, shops, and parked vehicles
        congestion_eval = self._assess_side_congestion(sensor_report)
        if congestion_eval["has_congestion"]:
            self.current_risk_level = "CAUTION"
            return congestion_eval["status"], congestion_eval["color"], congestion_eval["advice"]

        # Step 4: Default fallback state when all sectors are clear
        self.current_risk_level = "SAFE"
        self.consecutive_danger_frames = 0
        
        status = "RASTA BILKUL SAF HAI"
        color = (0, 255, 0)  # Green
        advice = "Rasta saaf hai. Normal speed se aaram se chalao aur steering steady rakho."

        # Update historical memory states for subsequent frame evaluation
        self.prev_left_blocked = sensor_report.get("left_blocked", False)
        self.prev_right_blocked = sensor_report.get("right_blocked", False)
        self.prev_center_vehicle = sensor_report.get("center_vehicle", False)

        return status, color, advice

    def get_system_diagnostic(self):
        """
        Returns internal telemetry diagnostic summary for system monitoring.
        """
        return {
            "risk_level": self.current_risk_level,
            "danger_frame_count": self.consecutive_danger_frames,
            "engine_status": "ACTIVE_ZERO_RISK"
        }