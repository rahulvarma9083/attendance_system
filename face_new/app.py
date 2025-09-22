import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime, date
import base64
from PIL import Image
import io
from deepface import DeepFace
import tempfile
import time

# Configure page
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            face_encoding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            status TEXT DEFAULT 'Present',
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Database operations
def add_student(student_id, name, face_encoding):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO students (student_id, name, face_encoding)
            VALUES (?, ?, ?)
        ''', (student_id, name, face_encoding))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_students():
    conn = sqlite3.connect('attendance.db')
    df = pd.read_sql_query('SELECT * FROM students', conn)
    conn.close()
    return df

def mark_attendance(student_id, confidence):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    current_date = date.today().isoformat()  # Convert to string format
    current_time = datetime.now().time().isoformat()  # Convert to string format
    
    # Check if already marked today
    cursor.execute('''
        SELECT * FROM attendance 
        WHERE student_id = ? AND date = ?
    ''', (student_id, current_date))
    
    if cursor.fetchone():
        conn.close()
        return False, "Already marked today"
    
    cursor.execute('''
        INSERT INTO attendance (student_id, date, time, confidence)
        VALUES (?, ?, ?, ?)
    ''', (student_id, current_date, current_time, confidence))
    
    conn.commit()
    conn.close()
    return True, "Attendance marked successfully"

def get_attendance_records(date_filter=None):
    conn = sqlite3.connect('attendance.db')
    query = '''
        SELECT a.student_id, s.name, a.date, a.time, a.status, a.confidence
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
    '''
    
    if date_filter:
        # Convert date_filter to string format if it's a date object
        if hasattr(date_filter, 'isoformat'):
            date_filter = date_filter.isoformat()
        query += ' WHERE a.date = ?'
        df = pd.read_sql_query(query, conn, params=(date_filter,))
    else:
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

# Face recognition functions
def extract_face_encoding(image):
    temp_path = None
    try:
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB for DeepFace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create temporary file with unique name
        temp_dir = tempfile.gettempdir()
        temp_filename = f"face_temp_{int(time.time())}_{np.random.randint(1000, 9999)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save image to temporary file (convert back to BGR for cv2.imwrite)
        if len(image.shape) == 3:
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            save_image = image
        cv2.imwrite(temp_path, save_image)
        
        # Extract embedding using DeepFace
        result = DeepFace.represent(
            img_path=temp_path,
            model_name='Facenet512',
            enforce_detection=True
        )
        
        embedding = result[0]['embedding']
        st.success(f"✅ Face detected and encoded successfully! (Embedding size: {len(embedding)})")
        return np.array(embedding)
        
    except Exception as e:
        st.error(f"Face encoding error: {str(e)}")
        st.info("💡 Tips for better face detection:")
        st.info("• Ensure good lighting")
        st.info("• Keep face clearly visible and centered")
        st.info("• Remove glasses/masks if possible")
        st.info("• Try taking photo from slightly different angle")
        return None
    finally:
        # Clean up temporary file with retry logic
        if temp_path and os.path.exists(temp_path):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    time.sleep(0.1)  # Small delay
                    os.unlink(temp_path)
                    break
                except (PermissionError, FileNotFoundError, OSError):
                    if attempt == max_retries - 1:
                        # If we can't delete after all retries, log it but don't fail
                        pass
                    else:
                        time.sleep(0.5)  # Wait longer before retry

def recognize_face(image, threshold=0.4):  # Reduced threshold for better matching
    try:
        # Get all students
        students = get_all_students()
        if students.empty:
            st.warning("No students registered yet!")
            return None, 0
        
        st.info(f"🔍 Comparing against {len(students)} registered students...")
        
        # Extract encoding from current image
        current_encoding = extract_face_encoding(image)
        if current_encoding is None:
            return None, 0
        
        best_match = None
        best_distance = float('inf')
        all_distances = []
        
        for idx, student in students.iterrows():
            if student['face_encoding']:
                try:
                    stored_encoding = np.frombuffer(student['face_encoding'], dtype=np.float64)
                    
                    # Use cosine distance for better face comparison
                    dot_product = np.dot(current_encoding, stored_encoding)
                    norm_a = np.linalg.norm(current_encoding)
                    norm_b = np.linalg.norm(stored_encoding)
                    cosine_similarity = dot_product / (norm_a * norm_b)
                    distance = 1 - cosine_similarity  # Convert to distance
                    
                    all_distances.append((student['name'], distance))
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = student
                        
                except Exception as e:
                    st.warning(f"Error comparing with {student['name']}: {str(e)}")
                    continue
        
        # Debug information
        st.info("🔍 **Recognition Debug Info:**")
        for name, dist in sorted(all_distances, key=lambda x: x[1])[:3]:  # Show top 3 matches
            st.info(f"• {name}: Distance = {dist:.4f}")
        
        if best_match is not None and best_distance < threshold:
            confidence = max(0, (threshold - best_distance) / threshold)
            st.success(f"✅ **Match Found!** {best_match['name']} (Distance: {best_distance:.4f}, Confidence: {confidence:.1%})")
            return best_match['student_id'], confidence
        else:
            st.error(f"❌ **No Match Found** (Best distance: {best_distance:.4f}, Threshold: {threshold})")
            st.info("💡 Try:")
            st.info("• Re-registering with a clearer photo")
            st.info("• Taking the attendance photo in similar lighting")
            st.info("• Ensuring face is clearly visible")
            return None, 0
            
    except Exception as e:
        st.error(f"Recognition error: {str(e)}")
        return None, 0

# Camera capture component with better mobile support
def camera_capture():
    st.markdown("📱 **Camera Access**")
    
    # Add camera troubleshooting info
    if st.button("❓ Camera Not Working?", help="Click for troubleshooting tips"):
        st.markdown("""
        **📱 Mobile Camera Troubleshooting:**
        
        **Step 1: Check URL**
        - URL must start with `https://` (not `http://`)
        - Streamlit Cloud provides HTTPS automatically
        
        **Step 2: Browser Permissions**
        - When prompted, click "Allow" for camera access
        - Check browser settings: Settings → Site permissions → Camera
        
        **Step 3: Try Different Browsers**
        - Chrome Mobile (recommended)
        - Firefox Mobile
        - Safari (iOS)
        - Edge Mobile
        
        **Step 4: Clear Browser Data**
        - Clear cache and cookies
        - Restart browser
        
        **Alternative: Upload Photo**
        - Take photo with camera app
        - Use file upload option below
        """)
    
    # Primary camera input
    captured_image = st.camera_input(
        "📸 Take Photo with Camera",
        help="Click to access your device camera",
        key="main_camera"
    )
    
    # Alternative file upload for cases where camera doesn't work
    st.markdown("**OR**")
    uploaded_image = st.file_uploader(
        "📁 Upload Photo from Gallery", 
        type=['jpg', 'jpeg', 'png'],
        help="Take photo with your camera app, then upload here"
    )
    
    # Process camera input
    if captured_image is not None:
        # Convert to OpenCV format
        image = Image.open(captured_image)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    # Process uploaded image
    elif uploaded_image is not None:
        # Convert to OpenCV format
        image = Image.open(uploaded_image)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    return None

# Streamlit UI
def main():
    init_database()
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .step-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196F3;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🎓 Smart Attendance System")
    st.markdown("**Automated attendance tracking using face recognition technology**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar navigation with better descriptions
    st.sidebar.title("🔧 Main Menu")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Select an option:",
        [
            "👤 Register New Student",
            "✅ Mark My Attendance", 
            "📊 View Attendance Records"
        ],
        index=0
    )
    
    # Add help section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📱 Mobile Camera Issues?")
    
    with st.sidebar.expander("🔧 Quick Fixes", expanded=True):
        st.markdown("""
        **✅ Check These:**
        1. URL starts with `https://`
        2. Allow camera permissions
        3. Use Chrome/Firefox browser
        4. Try refreshing the page
        
        **📁 Can't access camera?**
        Use the "Upload Photo" option instead!
        """)
    
    st.sidebar.markdown("### ❓ Need Help?")
    st.sidebar.markdown("""
    **For Students:**
    1. Register first with clear photo
    2. Mark attendance daily
    
    **For Teachers:**
    3. View all attendance records
    
    **📱 Mobile Tips:**
    - Grant camera permission when asked
    - Use good lighting
    - Keep face centered and clear
    """)
    
    if page == "👤 Register New Student":
        student_registration()
    elif page == "✅ Mark My Attendance":
        mark_attendance_page()
    elif page == "📊 View Attendance Records":
        teacher_dashboard()

def student_registration():
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("👤 New Student Registration")
    st.markdown("**Step 1:** Fill in your details and take a clear photo for registration")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📋 Registration Instructions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Before taking photo:**
            - Find good lighting
            - Remove glasses/hat if possible
            - Look directly at camera
            - Keep face centered
            """)
        with col2:
            st.markdown("""
            **Photo requirements:**
            - Clear face visibility
            - No shadows on face
            - Neutral expression
            - Front-facing position
            """)
    
    # Main registration form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Student Details")
        student_id = st.text_input(
            "Student ID *", 
            placeholder="e.g., STU001, 2021CS001",
            help="Enter your unique student ID"
        )
        student_name = st.text_input(
            "Full Name *", 
            placeholder="e.g., John Smith",
            help="Enter your complete name"
        )
        
        if student_id and student_name:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ Details entered successfully")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("📸 Take Your Photo")
        captured_image = camera_capture()
        
        if captured_image is not None:
            st.image(captured_image, caption="✅ Photo captured successfully", use_container_width=True)
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("📱 Click 'Take photo' button above to capture your image")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Registration button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        register_btn = st.button(
            "🚀 Complete Registration", 
            type="primary",
            use_container_width=True,
            disabled=not (student_id and student_name and captured_image is not None)
        )
    
    if register_btn:
        if student_id and student_name and captured_image is not None:
            with st.spinner("🔄 Processing your registration..."):
                face_encoding = extract_face_encoding(captured_image)
                
                if face_encoding is not None:
                    # Convert to bytes for storage
                    encoding_bytes = face_encoding.tobytes()
                    
                    if add_student(student_id, student_name, encoding_bytes):
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"### ✅ Registration Successful!")
                        st.markdown(f"**Welcome, {student_name}!**")
                        st.markdown(f"Your Student ID: **{student_id}**")
                        st.markdown("You can now mark attendance using the 'Mark My Attendance' option.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("⚠️ **Registration Failed**")
                        st.markdown("This Student ID is already registered. Please use a different ID.")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("❌ **Face Detection Failed**")
                    st.markdown("Could not detect your face clearly. Please try again with better lighting.")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show registered students
    st.markdown("---")
    with st.expander("👥 View All Registered Students"):
        students = get_all_students()
        if not students.empty:
            # Format the dataframe for better display
            display_df = students[['student_id', 'name', 'created_at']].copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                display_df.rename(columns={
                    'student_id': 'Student ID',
                    'name': 'Full Name',
                    'created_at': 'Registration Date'
                }),
                use_container_width=True,
                hide_index=True
            )
            st.info(f"📊 Total registered students: **{len(students)}**")
        else:
            st.info("No students have registered yet. Be the first one!")

def mark_attendance_page():
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("✅ Mark Your Attendance")
    st.markdown("**Step 2:** Take a photo to automatically mark your attendance")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if any students are registered
    students = get_all_students()
    if students.empty:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ **No Students Registered**")
        st.markdown("You need to register first before marking attendance.")
        st.markdown("👈 Go to 'Register New Student' to get started.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Mobile camera help - more prominent
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("📱 **Mobile Users:** If camera doesn't work, you can upload a photo taken with your camera app instead!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📱 How to Mark Attendance", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Method 1: Camera**
            1. Click 'Take Photo with Camera'
            2. Allow camera permissions
            3. Position face in camera view
            4. Take a clear photo
            """)
        with col2:
            st.markdown("""
            **Method 2: Upload Photo**
            1. Open camera app on phone
            2. Take a clear selfie
            3. Click 'Upload Photo from Gallery'
            4. Select the photo you just took
            """)
    
    # Main attendance marking interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Take or Upload Photo")
        captured_image = camera_capture()
    
    with col2:
        if captured_image is not None:
            st.subheader("🔍 Photo Ready")
            st.image(captured_image, caption="✅ Ready for recognition", use_container_width=True)
            
            # Recognition button
            recognize_btn = st.button(
                "🎯 Recognize & Mark Attendance", 
                type="primary",
                use_container_width=True
            )
            
            if recognize_btn:
                with st.spinner("🔄 Analyzing your photo..."):
                    student_id, confidence = recognize_face(captured_image)
                    
                    if student_id:
                        success, message = mark_attendance(student_id, confidence)
                        
                        if success:
                            # Get student name
                            student_name = students[students['student_id'] == student_id]['name'].iloc[0]
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"### ✅ Attendance Marked Successfully!")
                            st.markdown(f"**Student:** {student_name}")
                            st.markdown(f"**ID:** {student_id}")
                            st.markdown(f"**Time:** {datetime.now().strftime('%I:%M %p')}")
                            st.markdown(f"**Confidence:** {confidence:.1%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.markdown(f"⚠️ **{message}**")
                            st.markdown("You have already marked attendance for today.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("❌ **Face Not Recognized**")
                        st.markdown("We couldn't match your face with our records.")
                        st.markdown("**Possible solutions:**")
                        st.markdown("• Make sure you're registered")
                        st.markdown("• Try better lighting")
                        st.markdown("• Re-register if needed")
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.subheader("📱 How to Proceed")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            **Option 1: Use Camera**
            👈 Click 'Take Photo with Camera' on the left
            
            **Option 2: Upload Photo**
            👈 If camera doesn't work, use 'Upload Photo from Gallery'
            
            **🔧 Camera Issues?**
            - Make sure URL starts with `https://`
            - Allow camera permissions when prompted
            - Try Chrome or Firefox browser
            - Clear browser cache if needed
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Today's attendance summary
    st.markdown("---")
    st.subheader("📅 Today's Attendance Summary")
    
    today_attendance = get_attendance_records(date.today().isoformat())
    total_students = len(students)
    present_today = len(today_attendance)
    absent_today = total_students - present_today
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Students", total_students)
    with col2:
        st.metric("✅ Present Today", present_today)
    with col3:
        st.metric("❌ Absent Today", absent_today)
    with col4:
        if total_students > 0:
            attendance_rate = (present_today / total_students) * 100
            st.metric("📊 Attendance Rate", f"{attendance_rate:.1f}%")
        else:
            st.metric("📊 Attendance Rate", "0%")
    
    # Show today's attendance list
    if not today_attendance.empty:
        with st.expander(f"👥 View Today's Attendance List ({len(today_attendance)} students)", expanded=False):
            display_df = today_attendance.copy()
            display_df['time'] = pd.to_datetime(display_df['time']).dt.strftime('%I:%M %p')
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                display_df[['name', 'student_id', 'time', 'confidence']].rename(columns={
                    'name': 'Student Name',
                    'student_id': 'Student ID',
                    'time': 'Time Marked',
                    'confidence': 'Confidence'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("🔄 No attendance marked today yet. Be the first one!")

def teacher_dashboard():
    st.header("📊 Teacher Dashboard")
    
    # Summary statistics
    students = get_all_students()
    all_attendance = get_attendance_records()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Students", len(students))
    
    with col2:
        today_count = len(get_attendance_records(date.today().isoformat()))
        st.metric("✅ Today's Attendance", today_count)
    
    with col3:
        if not all_attendance.empty:
            avg_confidence = all_attendance['confidence'].mean()
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("🎯 Avg Confidence", "N/A")
    
    with col4:
        total_records = len(all_attendance)
        st.metric("📊 Total Records", total_records)
    
    # Date filter
    st.subheader("🔍 Filter Attendance")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date = st.date_input("Select Date", date.today())
    
    with col2:
        if st.button("📋 Show All Records"):
            selected_date = None
    
    # Attendance records
    if selected_date:
        st.subheader(f"📅 Attendance for {selected_date}")
        attendance_data = get_attendance_records(selected_date)
    else:
        st.subheader("📊 All Attendance Records")
        attendance_data = get_attendance_records()
    
    if not attendance_data.empty:
        # Format the dataframe
        display_df = attendance_data.copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df.rename(columns={
                'student_id': 'Student ID',
                'name': 'Name',
                'date': 'Date',
                'time': 'Time',
                'status': 'Status',
                'confidence': 'Confidence'
            }),
            use_container_width=True
        )
        
        # Download CSV
        csv = attendance_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"attendance_{selected_date or 'all'}.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found for the selected criteria.")
    
    # Attendance analytics
    if not all_attendance.empty:
        st.subheader("📈 Attendance Analytics")
        
        # Daily attendance chart
        daily_stats = all_attendance.groupby('date').size().reset_index(name='count')
        daily_stats['date'] = pd.to_datetime(daily_stats['date'])
        
        st.line_chart(daily_stats.set_index('date')['count'])

if __name__ == "__main__":
    main()