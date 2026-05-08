"""
Hospital AI Triage & Disease Classifier
=======================================
- Automated patient priority assignment
- 377 Symptom analysis
- Direct integration for FlexSim simulation data
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Hospital Scheduler",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS - Professional Clinical Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --accent: #00d4aa;
        --bg-card: #111827;
    }
    .main { background-color: #080c14; }
    .stButton>button {
        width: 100%;
        background-color: var(--accent);
        color: black;
        font-weight: bold;
        border-radius: 8px;
    }
    .metric-card {
        background-color: var(--bg-card);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid var(--accent);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Database Setup (For FlexSim Connection)
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect('hospital_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS triage_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_name TEXT,
                  age INTEGER,
                  predicted_disease TEXT,
                  priority_level INTEGER,
                  arrival_time TEXT)''')
    conn.commit()
    conn.close()

def save_prediction(name, age, disease, priority):
    conn = sqlite3.connect('hospital_data.db')
    c = conn.cursor()
    c.execute("INSERT INTO triage_results (patient_name, age, predicted_disease, priority_level, arrival_time) VALUES (?, ?, ?, ?, ?)",
              (name, age, disease, priority, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# Assets Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model = joblib.load('disease_classifier_model.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # 377 Validated Features
    features = [
        'anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 
        'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 
        'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 
        'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 
        'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 
        'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 
        'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 
        'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 
        'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 
        'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 
        'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 
        'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 
        'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 
        'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 
        'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 
        'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 
        'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 
        'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 
        'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 
        'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 
        'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 
        'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 
        'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 
        'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 
        'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 
        'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 
        'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 
        'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 
        'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 
        'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 
        'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 
        'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 
        'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 
        'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 
        'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 
        'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 
        'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 
        'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 
        'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 
        'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 
        'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 
        'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 
        'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 
        'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 
        'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 
        'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 
        'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 
        'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 
        'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 
        'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 
        'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 
        'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 
        'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 
        'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 
        'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 
        'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 
        'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 
        'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 
        'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 
        'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 
        'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 
        'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 
        'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 
        'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 
        'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 
        'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 
        'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 
        'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 
        'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 
        'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 
        'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 
        'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 
        'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 
        'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 
        'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 
        'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 
        'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 
        'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 
        'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 
        'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 
        'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 
        'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 
        'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 
        'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness'
    ]
    return model, le, features

# ─────────────────────────────────────────────────────────────────────────────
# Main Application Logic
# ─────────────────────────────────────────────────────────────────────────────
init_db()
model, le, feature_names = load_assets()

st.title("🏥 AI Hospital Admission & Triage System")
st.markdown("---")

# Sidebar for Navigation
menu = st.sidebar.selectbox("Navigate", ["Patient Admission", "Triage Analytics"])

if menu == "Patient Admission":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Patient Registration")
        p_name = st.text_input("Full Name")
        p_age = st.number_input("Age", min_value=0, max_value=120, value=25)
        
    with col2:
        st.subheader("🩺 Symptom Selection")
        selected_symptoms = st.multiselect(
            "Select symptoms presented by the patient:",
            options=feature_names
        )

    if st.button("Analyze & Assign Priority"):
        if p_name and selected_symptoms:
            # Prepare Input Vector
            input_vector = np.zeros(len(feature_names))
            for s in selected_symptoms:
                idx = feature_names.index(s)
                input_vector[idx] = 1
            
            # AI Prediction
            prediction_idx = model.predict(input_vector.reshape(1, -1))[0]
            disease = le.inverse_transform([prediction_idx])[0]
            
            # Priority Logic (Simplified example)
            critical_symptoms = ['sharp chest pain', 'shortness of breath', 'seizures', 'fainting']
            if any(s in selected_symptoms for s in critical_symptoms):
                priority = 1 # Immediate
                label = "Level 1: Critical (Immediate Action)"
                color = "red"
            elif len(selected_symptoms) > 5:
                priority = 2 # Urgent
                label = "Level 2: Urgent"
                color = "orange"
            else:
                priority = 3 # Non-Urgent
                label = "Level 3: Standard"
                color = "green"
            
            # Save to Database for FlexSim
            save_prediction(p_name, p_age, disease, priority)
            
            # Results Display
            st.success(f"Analysis Complete for {p_name}")
            res1, res2 = st.columns(2)
            res1.metric("Predicted Condition", disease)
            res2.metric("Assigned Priority", f"P{priority}")
            st.info(f"Guidance: {label}")
        else:
            st.warning("Please enter patient name and select at least one symptom.")

elif menu == "Triage Analytics":
    st.subheader("📊 Live Triage Dashboard")
    conn = sqlite3.connect('hospital_data.db')
    df = pd.read_sql_query("SELECT * FROM triage_results ORDER BY id DESC", conn)
    conn.close()
    
    if not df.empty:
        st.dataframe(df.style.highlight_max(axis=0, subset=['priority_level'], color='#3d1d1d'))
        
        # Priority Chart
        p_counts = df['priority_level'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=p_counts.index, values=p_counts.values, hole=.3)])
        fig.update_layout(title_text="Patient Priority Distribution", template="plotly_dark")
        st.plotly_chart(fig)
    else:
        st.write("No patient data recorded yet.")