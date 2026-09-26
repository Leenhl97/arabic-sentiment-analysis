import streamlit as st
import joblib
import re

# ============ إعداد الصفحة ============
st.set_page_config(
    page_title="تحليل مشاعر التقييمات",
    page_icon="🏨",
    layout="centered"
)

# ============ تصميم مخصص ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
    }

    .stApp {
        background-color: #0F1B2B;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 900;
        color: #F4E8D0;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        color: #8FA3B8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        direction: rtl;
        font-family: 'Tajawal', sans-serif;
        font-size: 1.1rem;
        background-color: #16273D;
        color: #F4E8D0;
        border: 1px solid #2A3F5A;
        border-radius: 10px;
    }

    .stButton button {
        background-color: #C9A15E;
        color: #0F1B2B;
        font-weight: 700;
        font-size: 1.05rem;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #DDB876;
        color: #0F1B2B;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-positive {
        background-color: rgba(90, 168, 130, 0.15);
        border: 1px solid #5AA882;
    }

    .result-negative {
        background-color: rgba(200, 92, 92, 0.15);
        border: 1px solid #C85C5C;
    }

    .result-label {
        font-size: 1.6rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }

    .confidence-text {
        color: #8FA3B8;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ============ تحميل النموذج ============
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ============ دوال التنظيف (نفس المستخدمة بالتدريب) ============
arabic_stopwords = [
    'من', 'في', 'على', 'الى', 'الي', 'عن', 'مع', 'هذا', 'هذه', 'ذلك',
    'التي', 'الذي', 'هو', 'هي', 'انا', 'انت', 'كان', 'كانت', 'يكون',
    'ان', 'او', 'ثم', 'ما', 'قد', 'كل', 'بعض',
    'هناك', 'هنا', 'الا', 'لكن', 'و', 'ف', 'ب', 'ل', 'ك'
]

def clean_text(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'[\u064B-\u0652\u0670\u0640]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in arabic_stopwords]
    return ' '.join(filtered)

def predict_sentiment(text):
    cleaned = remove_stopwords(clean_text(text))
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    confidence = probability[prediction] * 100
    return prediction, confidence

# ============ الواجهة ============
st.markdown('<div class="main-title">🏨 محلل مشاعر التقييمات</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">اكتبي تقييم فندق بالعربي، والنموذج بيحدد إذا كان إيجابي أو سلبي</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "نص التقييم",
    placeholder="مثال: الفندق نظيف والخدمة ممتازة والموظفين متعاونين...",
    height=140,
    label_visibility="collapsed"
)

# أمثلة جاهزة يقدر المستخدم يضغط عليها
st.markdown("<div style='color:#8FA3B8; font-size:0.9rem; margin-bottom:0.5rem;'>أو جربي أحد هاي الأمثلة:</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
example_clicked = None
with col1:
    if st.button("😊 مثال إيجابي", key="pos_ex"):
        example_clicked = "الفندق رائع والموقع ممتاز والغرفة نظيفة جدا"
with col2:
    if st.button("😞 مثال سلبي", key="neg_ex"):
        example_clicked = "الخدمة سيئة جدا والغرفة غير نظيفة ولا انصح احد يحجز هنا"

final_text = example_clicked if example_clicked else text_input

if st.button("حللي التقييم", type="primary") or example_clicked:
    if final_text.strip() == "":
        st.warning("اكتبي نص أول 🙂")
    else:
        prediction, confidence = predict_sentiment(final_text)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-positive">
                <div class="result-label" style="color:#5AA882;">تقييم إيجابي 😊</div>
                <div class="confidence-text">نسبة الثقة: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-negative">
                <div class="result-label" style="color:#C85C5C;">تقييم سلبي 😞</div>
                <div class="confidence-text">نسبة الثقة: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(confidence / 100)