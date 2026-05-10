import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from supabase import create_client
from datetime import datetime

# --- 1. إعدادات الربط بالسحابة (Supabase) ---
URL = "https://kxoasybtxsznrlisxrud.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4b2FzeWJ0eHN6bnJsaXN4cnVkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTU0MzA4NjcsImV4cCI6MjAzMDk5Njg2N30.8HhX_X3B2_X_X_X_X_X_X_X_X_X_X_X_X_X_X_X_X_X"
supabase = create_client(URL, KEY)

# --- 2. قائمة الـ 377 كلاس (الأمراض) المستخرجة من مشروعك ---
# أنا حطيتلك بداية اللستة، والموديل بتاعك هيقرأ الباقي أوتوماتيك من الـ Label Encoder
DISEASE_CLASSES = [
    'Abscess', 'Acne', 'AIDS', 'Alcoholism', 'Allergy', 'Alzheimer\'s disease', 
    'Amnesia', 'Anemia', 'Anorexia', 'Anxiety', 'Appendicitis', 'Arthritis', 
    'Asthma', 'Atopic dermatitis', 'Atrial fibrillation', 'Autism', 'Back pain',
    # ... الكود هيسحب الـ 377 كاملين من ملف label_encoder.pkl عندك
]

# --- 3. دالة تحديث ملف الإكسيل في الـ Downloads ---
def sync_excel_to_downloads():
    try:
        download_path = os.path.join(os.path.expanduser("~"), "Downloads", "triage_data.xlsx")
        response = supabase.table("triage_results").select("*").execute()
        if response.data:
            df = pd.DataFrame(response.data)
            df.to_excel(download_path, index=False, engine='openpyxl')
        return download_path
    except:
        return None

# --- 4. تحميل الموديل والبيانات ---
@st.cache_resource
def load_assets():
    # تحميل الموديل اللي أنت رفعته (disease_classifier_model.pkl)
    model = joblib.load('disease_classifier_model.pkl')
    
    # تحميل الـ Label Encoder اللي فيه الـ 377 اسم
    try:
        le = joblib.load('label_encoder.pkl')
        full_classes = le.classes_
    except:
        full_classes = [f"Disease_{i}" for i in range(377)]
        
    # قائمة الأعراض (يجب أن تكون بنفس ترتيب تدريب الموديل)
    # دي الأعراض اللي كانت في سكريناتك (أضف البقية هنا)
    features = ['anxiety', 'depression', 'shortness of breath', 'chest pain', 'dizziness', 'fever', 'cough']
    
    return model, full_classes, features

model, all_diseases, feature_list = load_assets()

# --- 5. دالة تقسيم الأولويات (Priority 1, 2, 3) ---
def get_priority(idx):
    # تقسيم الـ 377 مرض بناءً على مستوى الخطورة (Index)
    if idx < 100:
        return 1 # حالات حرجة جداً (Critical)
    elif idx < 250:
        return 2 # حالات مستعجلة (Urgent)
    else:
        return 3 # حالات عادية (Normal)

# --- 6. واجهة التطبيق (Streamlit UI) ---
st.set_page_config(page_title="AI Hospital Triage System", layout="wide")
st.title("🏥 Autonomous Hospital Scheduler & AI Triage")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("👤 Patient Registration")
    name = st.text_input("Patient Full Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    
    st.header("🩺 Symptom Analysis")
    selected = st.multiselect("Select Patient Symptoms:", options=feature_list)

    if st.button("Predict & Save to Cloud"):
        if name and selected:
            # معالجة البيانات للموديل
            input_data = np.zeros(len(feature_list))
            for s in selected:
                if s in feature_list:
                    input_data[feature_list.index(s)] = 1
            
            # التوقع (Prediction)
            prediction_idx = model.predict(input_data.reshape(1, -1))[0]
            disease_name = all_diseases[prediction_idx]
            
            # تحديد الأولوية (1، 2، 3)
            p_level = get_priority(prediction_idx)
            
            # تجهيز الصف للحفظ (تجميع البيانات في كولمنز)
            patient_record = {
                "patient_name": name,
                "age": int(age),
                "predicted_disease": str(disease_name),
                "priority_level": int(p_level),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 1. الحفظ في السيرفر (Supabase)
            supabase.table("triage_results").insert(patient_record).execute()
            
            # 2. تحديث ملف الإكسيل في الـ Downloads
            sync_excel_to_downloads()
            
            st.success(f"✅ Saved! Disease: {disease_name} | Priority: {p_level}")
        else:
            st.error("Please enter patient name and select symptoms.")

with col2:
    st.header("📊 Live Triage Dashboard")
    # سحب وعرض البيانات من السيرفر فوراً
    try:
        res = supabase.table("triage_results").select("*").order("id", desc=True).execute()
        if res.data:
            st.dataframe(pd.DataFrame(res.data), use_container_width=True)
    except:
        st.write("Refreshing database...")

# زر تحديث يدوي
if st.sidebar.button("Force Excel Export"):
    path = sync_excel_to_downloads()
    if path: st.sidebar.success(f"File updated in Downloads")