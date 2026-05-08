import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# 1. إعدادات الصفحة
st.set_page_config(page_title="Hospital AI Triage System", page_icon="🏥", layout="wide")

# تصميم بسيط للواجهة
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #00d4aa; color: black; font-weight: bold; border-radius: 10px; }
    .priority-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; border: 2px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الموديل والبيانات التعريفية
@st.cache_resource
def load_hospital_assets():
    # تحميل الموديل والـ Encoder (تأكد أن الملفات في نفس المجلد)
    model = joblib.load('disease_classifier_model.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # قائمة الـ 377 عرض بالترتيب الصحيح (منسقة من النوت بوك الخاص بك)
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
        'mass on vulvulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 
        'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 
        'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 
        'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 
        'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 
        'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 
        'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness'
    ]
    return model, le, feature_names
# تحميل الأصول (Assets)
try:
    model, le, feature_names = load_hospital_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. دالة تحديد الأولوية (لـ FlexSim)
def get_priority(disease_name):
    # تحويل لـ lowercase لضمان التطابق
    d = disease_name.lower()
    
    # حالات الطوارئ (الأولوية 1)
    emergency = ['heart attack', 'stroke', 'sepsis', 'cardiac', 'respiratory failure', 'unconscious']
    # حالات عاجلة (الأولوية 2)
    urgent = ['pneumonia', 'appendicitis', 'fracture', 'kidney failure', 'bleeding']
    
    if any(e in d for e in emergency): return 1
    if any(u in d for u in urgent): return 2
    return 3 # حالة عادية

# 4. واجهة التطبيق
st.title("🏥 نظام الفرز الطبي الذكي")
st.write("قم بإدخال بيانات المريض والأعراض لتحديد الأولوية وإرسال البيانات لـ FlexSim.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("بيانات المريض")
    name = st.text_input("اسم المريض")
    age = st.number_input("العمر", 0, 120, 25)
    arrival_time = st.time_input("وقت الوصول")

with col2:
    st.header("اختيار الأعراض")
    selected_symptoms = st.multiselect("اختر الأعراض (يمكنك البحث بكتابة اسم العرض):", feature_names)

if st.button("تحليل الحالة"):
    if not name or not selected_symptoms:
        st.error("يرجى إدخال الاسم واختيار الأعراض.")
    else:
        # بناء الـ Vector (377 عمود)
        input_data = np.zeros((1, len(feature_names)))
        for s in selected_symptoms:
            idx = feature_names.index(s)
            input_data[0, idx] = 1
        
        # التوقع
        prediction = model.predict(input_data)[0]
        disease = le.inverse_transform([prediction])[0]
        priority_val = get_priority(disease)
        
        # عرض النتيجة
        st.success(f"التشخيص المتوقع: {disease}")
        
        color = "#ff4b4b" if priority_val == 1 else "#ffa500" if priority_val == 2 else "#00d4aa"
        st.markdown(f"""
            <div class="priority-box" style="background-color: {color};">
                <h2 style="color: black;">مستوى الأولوية لـ FlexSim: {priority_val}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        # حفظ في قاعدة البيانات (لربطها بـ FlexSim)
        conn = sqlite3.connect('hospital_data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS triage 
                     (patient_name TEXT, age INTEGER, disease TEXT, priority INTEGER, time TEXT)''')
        c.execute("INSERT INTO triage VALUES (?, ?, ?, ?, ?)", 
                  (name, age, disease, priority_val, str(arrival_time)))
        conn.commit()
        conn.close()
        st.info("تم حفظ البيانات في hospital_data.db بنجاح.")

# عرض الطابور الحالي
if st.checkbox("عرض قائمة الانتظار الحالية"):
    try:
        conn = sqlite3.connect('hospital_data.db')
        df_display = pd.read_sql_query("SELECT * FROM triage ORDER BY priority ASC", conn)
        st.dataframe(df_display)
        conn.close()
    except:
        st.write("القائمة فارغة حالياً.")