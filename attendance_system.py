import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pickle

class FaceRecognitionAttendance:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.known_face_names = []
        self.known_face_ids = []
        self.attendance_file = 'attendance.csv'
        
        # Load or create face recognizer model
        self.model_file = 'face_model.yml'
        self.labels_file = 'labels.pkl'
        
        # Load employees
        self.load_employees()
        
        # Load attendance
        self.attendance_df = self.load_attendance()
    
    def load_employees(self):
        """Load and train face recognizer with employee images"""
        path = 'employees'
        
        if not os.path.exists(path):
            os.makedirs(path)
            print("No employees found. Please register employees first.")
            return
        
        faces = []
        labels = []
        label_dict = {}
        current_label = 0
        
        print("Loading and training faces...")
        
        for file in os.listdir(path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Extract employee info from filename (format: ID_Name.jpg)
                name_id = os.path.splitext(file)[0]
                try:
                    # Handle different filename formats
                    if '_' in name_id:
                        emp_id, name = name_id.split('_', 1)
                    else:
                        # If no underscore, use filename as name and generate ID
                        emp_id = str(current_label + 1000)
                        name = name_id
                    
                    # Read image
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {file}")
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Detect face
                    faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces_rect) == 0:
                        print(f"No face detected in {file}. Please use a clear face photo.")
                        continue
                    
                    for (x, y, w, h) in faces_rect:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        
                        faces.append(face_roi)
                        labels.append(current_label)
                        label_dict[current_label] = {'name': name, 'id': emp_id}
                        
                        self.known_face_names.append(name)
                        self.known_face_ids.append(emp_id)
                        
                        current_label += 1
                        print(f"  ✓ Loaded: {name} (ID: {emp_id})")
                        
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        
        if faces:
            # Train the recognizer
            self.face_recognizer.train(faces, np.array(labels))
            
            # Save the model
            self.face_recognizer.save(self.model_file)
            with open(self.labels_file, 'wb') as f:
                pickle.dump(label_dict, f)
            
            print(f"\n✓ Trained with {len(faces)} faces from {len(label_dict)} employees")
        else:
            print("\n✗ No valid faces found in employees directory")
            print("Please make sure:")
            print("  1. Images are clear with visible faces")
            print("  2. Images are in JPG or PNG format")
            print("  3. Filename format: ID_Name.jpg (e.g., 101_John.jpg)")
    
    def load_attendance(self):
        """Load or create attendance CSV file"""
        try:
            if os.path.exists(self.attendance_file):
                # Check if file is empty
                if os.path.getsize(self.attendance_file) == 0:
                    # Create new DataFrame with columns
                    df = pd.DataFrame(columns=['Employee_ID', 'Name', 'Date', 'Time'])
                    df.to_csv(self.attendance_file, index=False)
                    return df
                else:
                    # Try to read existing file
                    df = pd.read_csv(self.attendance_file)
                    # Ensure all required columns exist
                    required_columns = ['Employee_ID', 'Name', 'Date', 'Time']
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = ''
                    return df
            else:
                # Create new file with headers
                df = pd.DataFrame(columns=['Employee_ID', 'Name', 'Date', 'Time'])
                df.to_csv(self.attendance_file, index=False)
                print(f"Created new attendance file: {self.attendance_file}")
                return df
        except pd.errors.EmptyDataError:
            # File exists but is empty
            df = pd.DataFrame(columns=['Employee_ID', 'Name', 'Date', 'Time'])
            df.to_csv(self.attendance_file, index=False)
            print(f"Recreated empty attendance file: {self.attendance_file}")
            return df
        except Exception as e:
            print(f"Error loading attendance file: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(columns=['Employee_ID', 'Name', 'Date', 'Time'])
    
    def mark_attendance(self, emp_id, name):
        """Mark attendance for recognized employee"""
        today = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        try:
            # Check if already marked today
            if not self.attendance_df.empty:
                already_marked = self.attendance_df[
                    (self.attendance_df['Employee_ID'] == emp_id) & 
                    (self.attendance_df['Date'] == today)
                ]
                
                if not already_marked.empty:
                    print(f"  {name} already marked today at {already_marked.iloc[0]['Time']}")
                    return False
            
            # Add new attendance record
            new_record = pd.DataFrame({
                'Employee_ID': [emp_id],
                'Name': [name],
                'Date': [today],
                'Time': [current_time]
            })
            
            self.attendance_df = pd.concat([self.attendance_df, new_record], ignore_index=True)
            self.attendance_df.to_csv(self.attendance_file, index=False)
            print(f"  ✓ ATTENDANCE MARKED: {name} at {current_time}")
            return True
            
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def run_recognition(self):
        """Main face recognition loop"""
        if not self.known_face_names:
            print("\n" + "="*50)
            print("NO EMPLOYEES REGISTERED!")
            print("Please register employees first using register_employee.py")
            print("="*50)
            input("\nPress Enter to exit...")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            print("Please check:")
            print("  1. Camera is connected")
            print("  2. No other application is using the camera")
            print("  3. Camera drivers are installed")
            input("\nPress Enter to exit...")
            return
        
        # For FPS calculation
        fps_counter = 0
        fps_time = datetime.now()
        fps = 0
        
        # Track processed faces to avoid multiple marks (simple cooldown)
        processed_faces = {}  # emp_id -> last_mark_time
        
        print("\n" + "="*50)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        print(f"✓ Employees loaded: {len(self.known_face_names)}")
        print(f"✓ Camera initialized")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Calculate FPS
            fps_counter += 1
            if (datetime.now() - fps_time).seconds >= 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = datetime.now()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                # Predict
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    
                    # Lower confidence is better in LBPH (0 is perfect match)
                    if confidence < 80:  # Adjust threshold as needed
                        name = self.known_face_names[label]
                        emp_id = self.known_face_ids[label]
                        color = (0, 255, 0)  # Green for recognized
                        status = "RECOGNIZED"
                        
                        # Mark attendance (with cooldown to avoid multiple marks)
                        current_time = datetime.now()
                        if emp_id not in processed_faces or \
                           (current_time - processed_faces[emp_id]).total_seconds() > 60:  # 1 minute cooldown
                            self.mark_attendance(emp_id, name)
                            processed_faces[emp_id] = current_time
                    else:
                        name = "Unknown"
                        emp_id = ""
                        color = (0, 0, 255)  # Red for unknown
                        status = "UNKNOWN"
                        confidence = 100 - confidence  # Invert for display
                        
                except Exception as e:
                    name = "Unknown"
                    emp_id = ""
                    color = (0, 0, 255)
                    status = "ERROR"
                    confidence = 0
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw background for name
                cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                
                # Display name and ID
                if name != "Unknown":
                    display_text = f"{name} ({emp_id})"
                else:
                    display_text = "Unknown"
                
                cv2.putText(frame, display_text, (x+5, y-10), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
                # Show confidence
                if confidence > 0:
                    conf_text = f"{confidence:.0f}%"
                    conf_color = (0, 255, 0) if confidence < 80 else (0, 255, 255)
                    cv2.putText(frame, conf_text, (x+w-60, y-10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.4, conf_color, 1)
            
            # Display info panel
            info_y = 30
            cv2.putText(frame, f"Faces: {len(faces)}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Employees: {len(self.known_face_names)}", (10, info_y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {fps}", (10, info_y+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (frame.shape[1]-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show today's attendance count
            today = datetime.now().strftime('%Y-%m-%d')
            today_count = len(self.attendance_df[self.attendance_df['Date'] == today]) if not self.attendance_df.empty else 0
            cv2.putText(frame, f"Today: {today_count}", (frame.shape[1]-150, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the frame
            cv2.imshow('Face Recognition Attendance System', frame)
            
            # Check for keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'capture_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"  ✓ Frame saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        today = datetime.now().strftime('%Y-%m-%d')
        today_attendance = self.attendance_df[self.attendance_df['Date'] == today] if not self.attendance_df.empty else pd.DataFrame()
        print(f"Date: {today}")
        print(f"Total employees registered: {len(self.known_face_names)}")
        print(f"Present today: {len(today_attendance)}")
        if not today_attendance.empty:
            print("\nPresent employees:")
            for _, row in today_attendance.iterrows():
                print(f"  • {row['Name']} (ID: {row['Employee_ID']}) at {row['Time']}")
        print(f"\nAttendance saved to: {self.attendance_file}")
        print("="*50)

if __name__ == "__main__":
    try:
        system = FaceRecognitionAttendance()
        system.run_recognition()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have registered employees first (run register_employee.py)")
        print("2. Check that your camera is working")
        print("3. Ensure all required packages are installed:")
        print("   pip install opencv-python opencv-contrib-python numpy pandas")
        input("\nPress Enter to exit...")