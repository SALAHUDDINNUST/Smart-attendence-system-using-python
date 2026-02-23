import cv2
import os

def register_employee():
    """Register a new employee by capturing their face from multiple angles"""
    
    # Create employees directory if it doesn't exist
    if not os.path.exists('employees'):
        os.makedirs('employees')
        print("Created employees directory")
    
    # Get employee details
    print("\n" + "="*50)
    print("EMPLOYEE REGISTRATION")
    print("="*50)
    
    while True:
        emp_id = input("\nEnter Employee ID (numbers only): ").strip()
        if emp_id.isdigit():
            break
        else:
            print("Please enter numbers only!")
    
    name = input("Enter Employee Name: ").strip()
    if not name:
        name = f"Employee_{emp_id}"
    
    print(f"\nRegistering: {name} (ID: {emp_id})")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        input("Press Enter to exit...")
        return
    
    # Counter for captured photos
    photo_count = 0
    max_photos = 5  # Take multiple photos for better training
    
    print("\n📸 INSTRUCTIONS:")
    print("1. Look directly at the camera")
    print("2. Ensure good lighting")
    print("3. Move your face slightly for different angles")
    print("4. Press SPACE to capture each photo")
    print("5. Press ESC to cancel\n")
    
    print(f"📷 Need to capture {max_photos} photos for best results\n")
    
    while photo_count < max_photos:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw face guide - using rectangle instead of ellipse to avoid compatibility issues
        center_x = width // 2
        center_y = height // 2
        face_size = min(width, height) // 3
        
        # Draw a rectangle as face guide (more compatible)
        x1 = center_x - face_size
        y1 = center_y - face_size
        x2 = center_x + face_size
        y2 = center_y + face_size
        
        # Draw rectangle for face placement
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw crosshair in the center
        cv2.line(display_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(display_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        # Add text instructions
        cv2.putText(display_frame, "Place face in rectangle", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show progress
        progress_text = f"Photos: {photo_count}/{max_photos}"
        cv2.putText(display_frame, progress_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show employee info
        cv2.putText(display_frame, f"Name: {name}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"ID: {emp_id}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show angle suggestions
        angles = ["Front", "Left", "Right", "Up", "Down"]
        if photo_count < len(angles):
            angle_text = f"Next: {angles[photo_count]} angle"
            cv2.putText(display_frame, angle_text, (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(display_frame, "SPACE: Capture | ESC: Exit", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Register Employee', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 32:  # SPACE key
            # Save the full frame (will detect face during training)
            photo_count += 1
            filename = f"employees/{emp_id}_{name}_{photo_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  ✓ Photo {photo_count} captured: {filename}")
            
            # Show success message briefly
            success_frame = display_frame.copy()
            cv2.putText(success_frame, f"Photo {photo_count} Captured!", (center_x - 100, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('Register Employee', success_frame)
            cv2.waitKey(500)  # Show for 500ms
            
        elif key == 27:  # ESC key
            print("\nRegistration cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if photo_count == max_photos:
        print(f"\n✅ SUCCESS! Registered {name} with {photo_count} photos")
        print("You can now run the attendance system")
    elif photo_count > 0:
        print(f"\n⚠️ Registered {name} with {photo_count} photos (recommended: {max_photos})")
    else:
        print("\n❌ No photos captured")

if __name__ == "__main__":
    try:
        register_employee()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is connected and working")
        print("2. Close other applications that might be using the camera")
        print("3. Run: pip install opencv-python")
    
    input("\nPress Enter to exit...")