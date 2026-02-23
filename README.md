# Smart-attendence-system-using-python
This repo is of smart attendence system using python and its libraries (Tensor flow).As in real world the smart attendence system is very efficent , it save time.So this is a praticel exapmple where i train a model so that first we put his id and his name then the program run and then camera will become open and capture photo and use them for next time to mark the attendence.

💻 Usage Guide

1️⃣ Register New Employees

Registration Process:

Enter Employee ID (numbers only)
Enter Employee Name
Position face in the green rectangle
Press SPACE to capture 5 photos from different angles
Photos are saved in the employees folder

2️⃣ Run Attendance System

3️⃣ GUI Version (Optional)

4️⃣ Reset System

🎯 How It Works

Face Detection

Uses OpenCV's Haar Cascade classifier

Detects faces in real-time video stream

Draws rectangles around detected faces

Face Recognition

Employs LBPH (Local Binary Patterns Histograms) algorithm
Each employee is assigned a unique label
Confidence score determines recognition accuracy (lower = better)
Threshold: 80 (values below indicate good match)

📊 Output Examples
==================================================
FACE RECOGNITION ATTENDANCE SYSTEM
==================================================
✓ Employees loaded: 3

Registered Employees:
  1. Salah (ID: 101)
  2. Salah uddin (ID: 102)
  3. ALi (ID: 103)

✓ Camera initialized

Controls:
  'q' - Quit
  's' - Save current frame

  ✓ ATTENDANCE MARKED: Salah at 14:25:30
  ✓ ATTENDANCE MARKED: Ali at 14:26:15

  📈 Future Enhancements
Email Notifications: Send attendance reports via email
Cloud Integration: Store data in cloud database
Mobile App: Remote attendance monitoring

Developed by SALAH UDDIN
  
