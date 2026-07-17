import pandas as pd
import os
from datetime import datetime
from typing import Dict, List

class AttendanceDatabase:
    def __init__(self, records_folder: str):
        self.records_folder = records_folder
        self.current_attendance: List[Dict] = []
        os.makedirs(records_folder, exist_ok=True)
    
    def mark_attendance(self, student_name: str, timestamp: datetime):
        """Mark attendance for a student"""
        self.current_attendance.append({
            'Student': student_name,
            'Date': timestamp.strftime('%Y-%m-%d'),
            'Time': timestamp.strftime('%H:%M:%S')
            
        })
    
    def save_attendance(self, filename: str = None):
        """Save current attendance to CSV"""
        if not filename:
            today = datetime.now().strftime('%Y-%m-%d')
            filename = f"attendance_{today}.csv"
        
        filepath = os.path.join(self.records_folder, filename)
        
        if self.current_attendance:
            df = pd.DataFrame(self.current_attendance)
            file_exists = os.path.exists(filepath)

            df.to_csv(
              filepath,
              mode='a',              # 👈 append mode
              header=not file_exists, # 👈 header sirf first time
              index=False
        )

            print(f"💾 Appended {len(self.current_attendance)} records to {filepath}")

            self.current_attendance = []  # optional: clear memory
        else:
         print("ℹ️ No attendance records to save")
    
    def get_attendance_report(self, date: str = None) -> pd.DataFrame:
        """Get attendance report for a specific date"""
        if date:
            filename = f"attendance_{date}.csv"
        else:
            files = [f for f in os.listdir(self.records_folder) if f.endswith('.csv')]
            if files:
                filename = max(files)  # Most recent
            else:
                return pd.DataFrame()
        
        filepath = os.path.join(self.records_folder, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        return pd.DataFrame()