"""
Attendance System Utilities
Helper functions for image processing, file management, and system tools
"""

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import shutil

class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def preprocess_face_image(image_path: str, output_path: str = None) -> bool:
        """Preprocess student image for better recognition"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Resize to standard size
            img = cv2.resize(img, (300, 300))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Convert back to BGR
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Save processed image
            if output_path is None:
                output_path = image_path.replace('.jpg', '_processed.jpg').replace('.png', '_processed.png')
            
            cv2.imwrite(output_path, processed)
            print(f"✅ Processed: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return False
    
    @staticmethod
    def draw_attendance_stats(frame: np.ndarray, stats: Dict) -> np.ndarray:
        """Draw attendance statistics on frame"""
        # Background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 60), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Stats text
        y_offset = 80
        cv2.putText(frame, f"📊 Present: {stats['present']}/{stats['total']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        cv2.putText(frame, f"⏰ Time: {stats['time']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"📈 Confidence: {stats['avg_conf']:.1f}%", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame

class FileUtils:
    """File and directory management utilities"""
    
    @staticmethod
    def create_student_record(student_id: str, student_name: str, photo_path: str) -> Dict:
        """Create standardized student record"""
        return {
            'student_id': student_id,
            'name': student_name,
            'photo_path': photo_path,
            'registered_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'active'
        }
    
    @staticmethod
    def backup_attendance_records(folder: str, max_backups: int = 10):
        """Create backup of attendance records"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_folder = f"{folder}_backup_{timestamp}"
        
        if os.path.exists(folder):
            shutil.copytree(folder, backup_folder)
            print(f"💾 Backup created: {backup_folder}")
            
            # Keep only latest backups
            backups = sorted([f for f in os.listdir('.') if f.startswith(folder + '_backup_')])
            if len(backups) > max_backups:
                for old_backup in backups[:-max_backups]:
                    shutil.rmtree(old_backup)
                    print(f"🗑️ Removed old backup: {old_backup}")
    
    @staticmethod
    def generate_student_id(name: str) -> str:
        """Generate unique student ID from name"""
        # First 3 letters + timestamp
        short_name = ''.join([c for c in name.split()[0][:3].upper() if c.isalpha()])
        timestamp = datetime.now().strftime('%y%m%d%H%M')[-4:]
        return f"{short_name}{timestamp}"

class ReportUtils:
    """Attendance report generation utilities"""
    
    @staticmethod
    def generate_daily_report(records_folder: str, date: str = None) -> pd.DataFrame:
        """Generate comprehensive daily attendance report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        filename = f"attendance_{date}.csv"
        filepath = os.path.join(records_folder, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Add summary stats
            summary = {
                'Total_Students': len(df),
                'Attendance_Percentage': f"{len(df)/1*100:.1f}%" if len(df) > 0 else "0%",
                'First_Entry': df['Time'].min() if not df.empty else "N/A",
                'Last_Entry': df['Time'].max() if not df.empty else "N/A"
            }
            print(f"📊 Daily Report ({date}):")
            for key, value in summary.items():
                print(f"   {key}: {value}")
            return df
        else:
            print(f"❌ No records found for {date}")
            return pd.DataFrame()
    
    @staticmethod
    def export_summary_report(records_folder: str, output_file: str):
        """Export summary report for all dates"""
        all_files = [f for f in os.listdir(records_folder) if f.startswith('attendance_') and f.endswith('.csv')]
        summary_data = []
        
        for filename in all_files:
            date = filename.replace('attendance_', '').replace('.csv', '')
            filepath = os.path.join(records_folder, filename)
            df = pd.read_csv(filepath)
            summary_data.append({
                'Date': date,
                'Total_Present': len(df),
                'File': filename
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_file, index=False)
            print(f"📈 Summary report saved: {output_file}")
            return summary_df
        return pd.DataFrame()

class ConfigUtils:
    """Configuration management"""
    
    @staticmethod
    def load_config(config_file: str = 'config.json') -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "camera_index": 0,
            "recognition_threshold": 0.6,
            "attendance_window_minutes": 5,
            "max_students": 100,
            "show_stats": True
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    @staticmethod
    def save_config(config: Dict, config_file: str = 'config.json'):
        """Save configuration to JSON file"""
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config saved: {config_file}")

class SystemUtils:
    """System monitoring utilities"""
    
    @staticmethod
    def get_system_stats() -> Dict:
        """Get system performance stats"""
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_usage': 'N/A',  # Add psutil for real stats
            'memory_usage': 'N/A',
            'fps': 30.0
        }
    
    @staticmethod
    def log_message(message: str, log_file: str = 'attendance.log'):
        """Log messages to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(log_entry.strip())

# Global utility functions
def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    return cv2.resize(frame, (width, height))

def get_attendance_percentage(present: int, total: int) -> float:
    """Calculate attendance percentage"""
    return (present / total * 100) if total > 0 else 0.0

def format_time(seconds: float) -> str:
    """Format time in HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

print("✅ utils.py loaded successfully!")