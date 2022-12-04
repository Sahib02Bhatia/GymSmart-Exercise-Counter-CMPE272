import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import PoseModel as pm
import json
import requests  # pip install requests
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
from PIL import Image
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components  # Import Streamlit
from streamlit_echarts import st_echarts #pip install streamlit_echarts
import altair as alt



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://drive.google.com/file/d/1OR3M7m3GGUJZttsMth8dzKPm4pHe0H14/view?usp=share_link");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
    

#loading animation using url
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#Adding animation
st.title("SWEAT, SMILE & REPEAT \U0001F642")
lottie_hello = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_qmrxbp0w.json")
st_lottie(lottie_hello)

with st.expander(" Please read instructions before you start"):
        st.write("""

         For Personal Training:
         
         1. Click the button above to start personal training.
         2. 0 and 100 define the start and end position of exercise
         3. Stop training on top right.\n\n

         
         For counting the number of reps:
             
         1. Click the start counter button to start the app.
         2. Set the camera to see arms and chest.
         3. Start exercising and look at your posture on the screen.
         4. Reset counter for next rep.
         5. Click stop button to stop.\n

         Lets start...
        """)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#Add an expander to the app 

st.header("Lets Start...")


    
count = 0

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #4169E1;
    color:#ffffff;
    }
</style>""", unsafe_allow_html=True)

runT = st.button('_____Start Personal Training_____')
#Adding a sidebar to the app
st.sidebar.title(u"\U0001F3CB\U0001F3FB""GymSmart Exercise App")


#run = st.sidebar.checkbox('Check to start and uncheck to stop!')
with st.sidebar:
    selected = option_menu("Main Menu", ["Home",'About','Description','Instructions'], 
        icons=['house','display','bar-chart-steps','card-list'], menu_icon="cast", default_index=1)
    selected

if selected == "Home":
    st.session_state.sidebar_state = 'collapsed'

if selected == "About":
    st.sidebar.text("""
The Gymsmart  Exercise  is a web- 
application  that  provides  the
users  to  assist with  exercise
and  other   types  of  physical
training.  This application  can 
be  used  from  home  and  while
away. This app helps the user to 
keep motivated over  long period 
of time. Other than a laptop, no 
equipment is necessary.""")

if selected == "Description":
    st.sidebar.text("""
We have got the app for you!

In this app you will learn how to
train yourself  using  AI powered 
gym tracker using AI Powered Pose
estimation. We've leverage Media-
Pipe, Python to detect  different
posts  from a  webcam  feed. Then 
render the results  to the screen
using  OpenCV. As  part  of it we 
used web cam how to extract joint
coordinates  and  calculate joint
angles! 

Steps we made: 
1. Set up MediaPipe for Python 
2. Estimate the poses using  your 
   Webcam and OpenCV
3. Extract join  coordinates from
   the detected pose
4. Calculate angles between joint
   using Numpy and Trigonometry
5. Building AIPowered Gym Tracker
   to count your reps

This application mainly provides 
the user with two  features that 
a user can  start  his  personal 
training  and a  user  can  also 
count  the  repetations  that he 
has  made. The  below  given are 
the  respective  steps  how  the 
user can use these two features.""")

if selected == "Instructions":
    st.sidebar.text("""
Follow the below instructions.

For Personal Training:         
 1. Click  the  button above to 
    start personal training.
 2. 0 and 100 define  the start 
    and the end position of the 
    exercise.
 3. Stop training on top right.

 
For counting the number of reps:
 1. Click  on the start counter 
    button to start the app.
 2. Set the  camera to see arms 
    and chest.
 3. Start  exercising,  look at  
    your posture on  the screen.
 4. Reset counter for next rep.
 5. Click stop button to stop.""")



run = st.button("_____Start Counter (for reps)_____")

FRAME_WINDOW = st.image([])





# Calculate Angles

def calculate_angle(a,b,c):
    # Reduce 3D point to 2D
    a = np.array([a.x, a.y])#, a.z])    
    b = np.array([b.x, b.y])#, b.z])
    c = np.array([c.x, c.y])#, c.z])  

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)
    
    # A.B = |A||B|cos(x) 
    theta = np.arccos(np.dot(ab, bc) / np.multiply(np.linalg.norm(ab), np.linalg.norm(bc)))     
    # Convert to degrees
    theta = 180 - 180 * theta / 3.14   
    return np.round(theta, 2)


def curlCOunter():
    # Connecting Keypoints Visuals
    mp_drawing = mp.solutions.drawing_utils     

    # Keypoint detection model
    mp_pose = mp.solutions.pose     

    # Flag which stores hand position(Either UP or DOWN)
    left_flag = None     
    right_flag = None

    # Storage for count of bicep curls
    right_count = 0
    left_count = 0       

    cap = cv2.VideoCapture(0)
    # Landmark detection model instance
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) 
    while cap.isOpened():
        _, frame = cap.read()

         # Convert BGR frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     
        image.flags.writeable = False
        
        # Make Detections
        # Get landmarks of the object in frame from the model
        results = pose.process(image)   

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of left part
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Get coordinates of right part
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate and get angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)      
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image,\
                    str(left_angle), \
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
            cv2.putText(image,\
                    str(right_angle), \
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640,480]).astype(int)),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2,cv2.LINE_AA)
        
            # Counter 
            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag=='down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag=='down':
                right_count += 1
                right_flag = 'up'
            
        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0,0), (1024,73), (10,10,10), -1)
        cv2.putText(image, 'Left=' + str(left_count) + '    Right=' + str(right_count),
                          (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #cv2.imshow('MediaPipe feed', image)

        FRAME_WINDOW.image(image)

        # Esc for quiting the app
        k = cv2.waitKey(30) & 0xff  
        if k==27:
            break
        elif k==ord('r'):       
            # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()


def startTrainer():
    cap = cv2.VideoCapture(0)
     
    detector = pm.poseDetector()
    count = 0
    dir = 0
    pTime = 0
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (1280, 720))
        # img = cv2.imread("AiTrainer/test.jpg")
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        # print(lmList)
        if len(lmList) != 0:
            # Right Arm
            angle = detector.findAngle(img, 12, 14, 16)
            # # Left Arm
            #angle = detector.findAngle(img, 11, 13, 15,False)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            # print(angle, per)
     
            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            print(count)
            
     
            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
     
            # Draw Curl Count
            #cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (25, 670), cv2.FONT_HERSHEY_PLAIN, 10,
                        (255, 0, 0), 15)
     
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'FPS:'+str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)
     
       # cv2.imshow("Image", img)
        
        # Render Detections
       
        #cv2.imshow('MediaPipe feed', image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img)
        cv2.waitKey(1)


if __name__ == '__main__':
    while run:
        reset = st.button('Stop')
        curlCOunter()
    while runT:
        reset = st.button('Stop')
        if(reset):
            count = 0
        startTrainer()
    else:
        st.write('Stopped')


st.write("""
\n
\n
\n

    """)




st.header("Diet plan set for health and nutrition")
options = {
    "title": {"text": "1. Maintainence", "subtext": "Moderate Protein\nModerate Carb\nModerate Fat", "left": "left"},
    "tooltip": {"trigger": "item"},
    "legend": {"orient": "vertical", "right": "right",},
    "series": [
        {
            "name": "Intake level Percent",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": 50, "name": "CARB"},
                {"value": 25, "name": "FAT"},
                {"value": 25, "name": "PROTEIN"},
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
st_echarts(
    options=options, height="500px",
)


options = {
    "title": {"text": "2. Weight Loss", "subtext": "High Protein\nLow Carb\nLow Fat", "left": "left"},
    "tooltip": {"trigger": "item"},
    "legend": {"orient": "horizontal", "right": "right",},
    "series": [
        {
            "name": "Intake level Percent",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": 45, "name": "CARB"},
                {"value": 20, "name": "FAT"},
                {"value": 35, "name": "PROTEIN"},
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
st_echarts(
    options=options, height="500px",
)

options = {
    "title": {"text": "3. Muscle Gain", "subtext": "High Protein\nLow Carb\nLow Fat", "left": "left"},
    "tooltip": {"trigger": "item"},
    "legend": {"orient": "horizontal", "right": "right",},
    "series": [
        {
            "name": "Intake level Percent",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": 45, "name": "CARB"},
                {"value": 15, "name": "FAT"},
                {"value": 40, "name": "PROTEIN"},
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
st_echarts(
    options=options, height="500px",
)






Food_1, Food_2, Food_3 = st.tabs(["Proteins", "Carbohydrates", "Fats"])

with Food_1:
    st.image("https://i.pinimg.com/originals/6a/e7/ce/6ae7ce626818f355336f8ddc10ad8c83.jpg")

with Food_2:
    col4,col5=st.columns(2)
    with col4:
        st.image("https://static.onecms.io/wp-content/uploads/sites/44/2018/05/27200157/vegetables-carbs-chart.png")

    with col5:
       st.image("https://www.justchartit.com/wp-content/uploads/2021/07/high-carb-vegetables-chart.jpg")

with Food_3:
    st.image("https://mlhiibtwx79w.i.optimole.com/aQUVB-g-7UefJFGh/w:1024/h:951/q:auto/https://mydietitian.com.au/wp-content/uploads/2020/09/The-Bad-Fats.png")



st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)


st.header("Fitness in a pie chart")
options = {
    "title": {"text": "In Percentage(%)", "subtext": "Stay Fit", "left": "center"},
    "tooltip": {"trigger": "item"},
    "legend": {"orient": "vertical", "left": "left",},
    "series": [
        {
            "name": "Percentage",
            "type": "pie",
            "radius": "80%",
            "data": [
                {"value": 20, "name": "Training"},
                {"value": 30, "name": "Nutrition"},
                {"value": 10, "name": "Patience"},
                {"value": 40, "name": "Kindness to Yourself"},
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
a1=st_echarts(
    options=options, height="500px",
    )


choose=st.radio("How was your experience?",("Poor","Satisfactory","Good","Very Good"))




    
