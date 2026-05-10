"""
Hospital AI Triage & Disease Classifier
=======================================
- Automated patient priority assignment
- 377 Symptom analysis
- Direct integration for FlexSim simulation data
"""
import streamlit as st
import pandas as pd
import joblib
from supabase import create_client, Client

# --- 1. إعدادات الصفحة (Page Configuration) ---
st.set_page_config(
    page_title="AI Hospital Scheduler",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. إعدادات الربط مع السحابة (Supabase Connection) ---
# ملاحظة: تأكد أن هذه القيم مطابقة تماماً لما في حسابك
SUPABASE_URL = "https://kxoasyhtxsznelisxrud.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4b2FzeWJ0eHN6bnJsaXN4cnVkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgzNTI5NjksImV4cCI6MjA5MzkyODk2OX0._f1FFz9vdoGecazw1Ta6wVPxAlskhZkB7K9IX0FPb0k"
# إنشاء كائن الاتصال (Client)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to connect to Supabase: {e}")

# --- 3. تحميل موديلات الذكاء الاصطناعي (AI Assets) ---
@st.cache_resource
def load_assets():
    try:
        # تحميل الموديل والـ Encoder
        # هام: يجب رفع هذه الملفات بجانب app.py على GitHub
        model = joblib.load('disease_classifier_model.pkl')
        le = joblib.load('label_encoder.pkl')
        
        # قائمة الخصائص (Features) - الـ 377 عرضاً كما في كودك
        feature_names = [
            'anxiety and nervousness', 'depression', 'shortness of breath',
            # ... (بقية الـ 377 عرضاً هنا)
        ]
        return model, le, feature_names
    except FileNotFoundError:
        st.error("Error: Model files (.pkl) not found. Please upload them to GitHub.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

# استدعاء التحميل
model, le, feature_names = load_assets()

# --- 4. دالة جلب البيانات وعرضها ---
def display_data():
    try:
        # سحب البيانات من جدول triage_results
        response = supabase.table("triage_results").select(
            "patient_name, age, predicted_disease, priority_level"
        ).execute()
        
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            st.subheader("📋 Patient Triage Records")
            st.dataframe(
                df, 
                use_container_width=True,
                column_config={
                    "patient_name": "Patient Name",
                    "age": "Age",
                    "predicted_disease": "Diagnosis",
                    "priority_level": "Priority"
                }
            )
            
            # إحصائيات سريعة أسفل الجدول
            col1, col2 = st.columns(2)
            col1.metric("Total Patients", len(df))
            high_count = len(df[df['priority_level'].str.lower() == 'high']) if 'priority_level' in df.columns else 0
            col2.metric("High Priority Cases", high_count)
    
    # 377 Validated Features
    feature_names =   [
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
    return model, le, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# Main Application Logic
# ─────────────────────────────────────────────────────────────────────────────
model, le, feature_names = load_assets()

# --- 4. دوال التعامل مع السحاب (Cloud Functions) ---
def save_prediction(name, age, disease, priority):
    try:
        data = {
            "patient_name": name,
            "age": int(age),
            "predicted_disease": disease,
            "priority_level": int(priority)
        }
        supabase.table("triage_results").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Cloud Error: {e}")
        return False

# --- 5. واجهة المستخدم الرئيسية (Main Application Logic) ---
st.title("🏥 AI Hospital Admission & Triage System")
st.markdown("---")

menu = st.sidebar.selectbox("Navigate", ["Patient Admission", "Triage Analytics"])

if menu == "Patient Admission":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Patient Registration")
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
            # تجهيز مصفوفة الأعراض للموديل
            input_vector = np.zeros(len(feature_names))
            for s in selected_symptoms:
                if s in feature_names:
                    idx = feature_names.index(s)
                    input_vector[idx] = 1
            
            # توقع المرض
            prediction_idx = model.predict(input_vector.reshape(1, -1))[0]
            disease = le.inverse_transform([prediction_idx])[0]

            # منطق الأولوية (Priority Logic)
            critical_symptoms = ['sharp chest pain', 'shortness of breath', 'seizures', 'fainting']
            if any(s in selected_symptoms for s in critical_symptoms):
                priority = 1
                label = "Level 1: Critical (Immediate Action)"
                color = "red"
            elif len(selected_symptoms) > 5:
                priority = 2
                label = "Level 2: Urgent"
                color = "orange"
            else:
                priority = 3
                label = "Level 3: Standard"
                color = "green"

            # حفظ في السحاب
            if save_prediction(p_name, p_age, disease, priority):
                st.success(f"Analysis Complete for {p_name}")
                
                res1, res2 = st.columns(2)
                res1.metric("Predicted Condition", disease)
                res2.metric("Assigned Priority", f"P{priority}")
                st.info(f"Guidance: {label}")
        else:
            st.warning("Please enter patient name and select at least one symptom.")

elif menu == "Triage Analytics":
    st.subheader("📊 Live Triage Dashboard")
    
    try:
        # سحب البيانات من السحاب
        response = supabase.table("triage_results").select("*").order("id", desc=True).execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            st.dataframe(df.style.highlight_max(axis=0, subset=['priority_level'], color='#3d1d1d'))

            # رسم بياني للأولويات
            p_counts = df['priority_level'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=p_counts.index, values=p_counts.values, hole=.3)])
            fig.update_layout(title_text="Patient Priority Distribution", template="plotly_dark")
            st.plotly_chart(fig)
        else:
            st.write("No patient data recorded yet.")
    except Exception as e:
        st.error(f"Error fetching data from cloud: {e}")