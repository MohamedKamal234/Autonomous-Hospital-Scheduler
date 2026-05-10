import streamlit as st
import pandas as pd
import joblib
import numpy as np
from supabase import create_client, Client

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Hospital Scheduler",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Supabase Connection (Updated with your new credentials) ---
# تم تصحيح الرابط بناءً على الـ Token الجديد
SUPABASE_URL = "https://kxoasybtxsznrlisxrud.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4b2FzeWJ0eHN6bnJsaXN4cnVkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgzNTI5NjksImV4cCI6MjA5MzkyODk2OX0._f1FFz9vdoGecazw1Ta6wVPxAlskhZkB7K9IX0FPb0k"

@st.cache_resource
def init_connection():
    try:
        # استخدام .strip() لإزالة أي مسافات مخفية قد تسبب خطأ في الاتصال
        return create_client(SUPABASE_URL.strip(), SUPABASE_KEY.strip())
    except Exception as e:
        st.error(f"Cloud Connection Failed: {e}")
        return None

supabase = init_connection()

# --- 3. AI Assets Loading ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('disease_classifier_model.pkl')
        le = joblib.load('label_encoder.pkl')
        # قائمة الـ 377 عرض بالترتيب الصحيح للموديل
        feature_names = [
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
            'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg looks infected', 'allergic reaction',
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
            'irregular appearing nails', 'itching of skin', 'hurts to breathe', 'nailbiting',
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
        return model, le, feature_names
    except Exception as e:
        st.error(f"Error loading AI assets: {e}")
        return None, None, None

model, le, feature_names = load_assets()

# --- 4. Cloud Logic ---
def save_patient_data(name, age, disease, priority):
    if not supabase: return False
    try:
        payload = {
            "patient_name": str(name),
            "age": int(age),
            "predicted_disease": str(disease),
            "priority_level": int(priority)
        }
        supabase.table("triage_results").insert(payload).execute()
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False

# --- 5. User Interface ---
st.title("🏥 AI Hospital Admission & Triage System")
st.markdown("---")

tab_admit, tab_view = st.tabs(["Patient Admission", "Triage Records"])

with tab_admit:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📝 Registration")
        name = st.text_input("Patient Full Name")
        age = st.number_input("Patient Age", 0, 120, 25)
    with c2:
        st.subheader("🔍 Symptoms")
        symptoms = st.multiselect("Select symptoms:", options=feature_names if feature_names else [])

    if st.button("Analyze & Save", use_container_width=True):
        if name and symptoms and model:
            # 1. Vectorization
            vec = np.zeros(len(feature_names))
            for s in symptoms:
                if s in feature_names: vec[feature_names.index(s)] = 1
            
            # 2. AI Prediction
            pred_idx = model.predict(vec.reshape(1, -1))[0]
            disease = le.inverse_transform([pred_idx])[0]
            
            # 3. Priority Logic
            critical = ['sharp chest pain', 'shortness of breath', 'seizures', 'fainting']
            if any(s in symptoms for s in critical):
                prio, label = 1, "Level 1: Critical"
            elif len(symptoms) > 5:
                prio, label = 2, "Level 2: Urgent"
            else:
                prio, label = 3, "Level 3: Standard"
            
            # 4. Save to Supabase
            if save_patient_data(name, age, disease, prio):
                st.success(f"Successfully processed: {name}")
                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Disease Prediction", disease)
                res_c2.metric("Priority Level", f"P{prio}")
                st.info(f"Clinical Guidance: {label}")
        else:
            st.warning("Please ensure name and symptoms are provided.")

with tab_view:
    st.subheader("📊 Live Triage Dashboard")
    if st.button("🔄 Refresh"): st.rerun()
    
    if supabase:
        try:
            query = supabase.table("triage_results").select("*").order("id", desc=True).execute()
            data_df = pd.DataFrame(query.data)
            if not data_df.empty:
                st.dataframe(data_df, use_container_width=True)
            else:
                st.info("No data in cloud yet.")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")