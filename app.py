import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from supabase import create_client, Client

# --- 1. إعدادات الصفحة ---
st.set_page_config(
    page_title="AI Hospital Scheduler",
    page_icon="🏥",
    layout="wide"
)

# --- 2. المسارات والبيانات (عدل المسار هنا لمسار جهازك) ---
LOCAL_EXCEL_PATH = r"C:\Users\DeLL\Downloads\Graduation project\patient_data.xlsx"

SUPABASE_URL = "https://kxoasybtxsznrlisxrud.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4b2FzeWJ0eHN6bnJsaXN4cnVkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgzNTI5NjksImV4cCI6MjA5MzkyODk2OX0._f1FFz9vdoGecazw1Ta6wVPxAlskhZkB7K9IX0FPb0k"

# --- 3. تهيئة الاتصال والموديل ---
@st.cache_resource
def init_connection():
    try:
        return create_client(SUPABASE_URL.strip(), SUPABASE_KEY.strip())
    except: return None

supabase = init_connection()

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('disease_classifier_model.pkl')
        le = joblib.load('label_encoder.pkl')
        # قائمة الـ 377 عرض (تم اختصارها هنا للعرض، تأكد من وجودها كاملة في ملفك)
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
    except: return None, None, None

model, le, feature_names = load_assets()

# --- 4. دوال الحفظ ---
def save_dual_mode(name, age, disease, priority):
    record = {
        "patient_name": name,
        "age": int(age),
        "predicted_disease": disease,
        "priority_level": int(priority)
    }
    
    # 1. الحفظ في السحاب
    cloud_success = False
    if supabase:
        try:
            supabase.table("triage_results").insert(record).execute()
            cloud_success = True
        except: pass

    # 2. الحفظ في الإكسل اللوكال
    local_success = False
    try:
        if os.path.exists(LOCAL_EXCEL_PATH):
            df = pd.read_excel(LOCAL_EXCEL_PATH)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            df = pd.DataFrame([record])
        df.to_excel(LOCAL_EXCEL_PATH, index=False)
        local_success = True
    except: pass

    return cloud_success, local_success

# --- 5. واجهة المستخدم ---
st.title("🏥 Smart Hospital Triage & Sync")
st.markdown(f"**Local Excel Path:** `{LOCAL_EXCEL_PATH}`")

tab1, tab2 = st.tabs(["Patient Entry", "Live Dashboard"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Patient Name")
        p_age = st.number_input("Age", 0, 120, 25)
    with col2:
        selected = st.multiselect("Symptoms", options=feature_names if feature_names else [])

    if st.button("Analyze & Save Everywhere", use_container_width=True):
        if p_name and selected and model:
            # Vectorization
            vec = np.zeros(len(feature_names))
            for s in selected:
                if s in feature_names: vec[feature_names.index(s)] = 1
            
            # Prediction
            res = model.predict(vec.reshape(1, -1))[0]
            disease = le.inverse_transform([res])[0]
            
            # Priority
            critical = ['sharp chest pain', 'shortness of breath', 'seizures', 'fainting']
            prio = 1 if any(s in selected for s in critical) else (2 if len(selected) > 5 else 3)
            
            # Saving
            c_ok, l_ok = save_dual_mode(p_name, p_age, disease, prio)
            
            if l_ok: st.success("✅ تم تحديث ملف الإكسل بنجاح")
            if c_ok: st.info("☁️ تم الرفع للسحاب")
            
            st.divider()
            r1, r2 = st.columns(2)
            r1.metric("Predicted Condition", disease)
            r2.metric("Triage Priority", f"P{prio}")
        else:
            st.error("Missing Data or Model Assets!")

with tab2:
    if st.button("🔄 Refresh View"): st.rerun()
    if supabase:
        try:
            data = supabase.table("triage_results").select("*").order("id", desc=True).execute()
            st.dataframe(pd.DataFrame(data.data), use_container_width=True)
        except: st.warning("Cloud data temporarily unavailable.")