import streamlit as st
import torch
from PIL import Image
import os
import sys
import pandas as pd
import plotly.express as px

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pipeline import AgroInferencePipeline

# Настройка страницы
st.set_page_config(
    page_title="AgroScan AI | Premium Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Словарь перевода болезней
DISEASE_TRANSLATIONS = {
    "RU": {
        "healthy": "Здоров", "Blight": "Фитофтороз", "Rust": "Ржавчина", "Spot": "Пятнистость",
        "Mold": "Плесень", "Virus": "Вирус", "Mites": "Клещи", "Scab": "Парша", "Rot": "Гниль",
        "Corn": "Кукуруза", "Tomato": "Томат", "Potato": "Картофель", "Pepper": "Перец",
        "Rice": "Рис", "Apple": "Яблоко", "Grape": "Виноград", "Peach": "Персик",
        "Strawberry": "Клубника", "Orange": "Апельсин", "Blueberry": "Черника", "Cherry": "Вишня",
        "maize": "", "common": "Обычная", "leaf": "листа", "Northern": "Северный", "Cercospora": "Церкоспороз",
        "Gray": "Серая"
    },
    "KZ": {
        "healthy": "Сау", "Blight": "Фитофтороз", "Rust": "Тот", "Spot": "Дақ",
        "Mold": "Зең", "Virus": "Вирус", "Mites": "Кенелер", "Scab": "Қотыр", "Rot": "Шірік",
        "Corn": "Жүгері", "Tomato": "Қызанақ", "Potato": "Картоп", "Pepper": "Бұрыш",
        "Rice": "Күріш", "Apple": "Алма", "Grape": "Жүзім", "Peach": "Шаттық",
        "Strawberry": "Құлпынай", "Orange": "Апельсин", "Blueberry": "Көкжидек", "Cherry": "Шие",
        "maize": "", "common": "Кәдімгі", "leaf": "жапырақ"
    }
}

def translate_disease(name, lang):
    if lang == "EN": return name
    
    # Очистка от лишних символов
    clean_name = name.replace("_", " ").replace("(", " ").replace(")", " ")
    words = clean_name.split()
    translated_words = []
    
    dict_lang = DISEASE_TRANSLATIONS.get(lang, {})
    
    for word in words:
        found = False
        for key, val in dict_lang.items():
            if key.lower() == word.lower():
                if val: translated_words.append(val)
                found = True
                break
        if not found:
            # Если точного совпадения нет, проверяем вхождение
            for key, val in dict_lang.items():
                if key.lower() in word.lower():
                    if val: translated_words.append(val)
                    found = True
                    break
        if not found:
            translated_words.append(word)
            
    return " ".join(translated_words).strip().replace("  ", " ")

# Кастомный CSS для МАКСИМАЛЬНОЙ ЧИТАЕМОСТИ
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');

    /* ГЛОБАЛЬНОЕ ПРИНУЖДЕНИЕ ЦВЕТА */
    html, body, [data-testid="stAppViewContainer"] {
        color: #000000 !important;
    }

    .stApp {
        background: #f8faf8 !important;
    }

    /* Принудительно черный цвет для ВСЕХ заголовков и текста */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #0d1f14 !important;
        opacity: 1 !important;
    }

    /* Исправление Рекомендаций (Warning box) */
    [data-testid="stNotification"] {
        background-color: #fff3cd !important;
        border: 2px solid #ffeeba !important;
        border-radius: 15px !important;
    }
    [data-testid="stNotification"] * {
        color: #856404 !important;
        font-weight: 700 !important;
    }

    /* Сайдбар */
    [data-testid="stSidebar"] {
        background: #112d1c !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Карточка результата */
    .prediction-card {
        background: white !important;
        padding: 30px;
        border-radius: 20px;
        border: 3px solid #1b4332;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .prediction-card h2 {
        color: #1b4332 !important;
        font-size: 2.5em !important;
    }

    /* Кнопки */
    .stButton>button {
        background: #1b4332 !important;
        border-radius: 15px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Переводы интерфейса
LANGUAGES = {
    "RU": {
        "title": "🌿 AgroScan AI: Диагностика",
        "subtitle": "Система компьютерного зрения для агрокомплекса",
        "sidebar_info": "Загрузите фото листа для мгновенного анализа.",
        "culture_filter": "Фильтр культуры",
        "upload_header": "### 📤 Загрузка изображения",
        "upload_btn": "🚀 Начать анализ",
        "results_header": "### 📊 Результаты анализа",
        "confidence": "Вероятность",
        "recommendation": "#### 💡 Рекомендация",
        "prob_dist": "#### Распределение вероятностей",
        "class_col": "Класс",
        "prob_col": "Вероятность",
        "placeholder": "Загрузите фото и нажмите 'Начать анализ'",
        "loading": "Анализируем структуру листа...",
        "success": "Анализ завершен!",
        "lang_select": "Язык / Til / Language"
    },
    "EN": {
        "title": "🌿 AgroScan AI: Diagnostics",
        "subtitle": "Computer vision system for agriculture",
        "sidebar_info": "Upload a leaf photo for instant analysis.",
        "culture_filter": "Culture Filter",
        "upload_header": "### 📤 Upload Image",
        "upload_btn": "🚀 Start Analysis",
        "results_header": "### 📊 Analysis Results",
        "confidence": "Confidence",
        "recommendation": "#### 💡 Recommendation",
        "prob_dist": "#### Probability Distribution",
        "class_col": "Class",
        "prob_col": "Probability",
        "placeholder": "Upload a photo and click 'Start Analysis'",
        "loading": "Analyzing leaf structure...",
        "success": "Analysis completed!",
        "lang_select": "Language"
    },
    "KZ": {
        "title": "🌿 AgroScan AI: Диагностика",
        "subtitle": "Агроөнеркәсіп кешеніне арналған компьютерлік көру жүйесі",
        "sidebar_info": "Жедел талдау үшін жапырақ фотосуретін жүктеңіз.",
        "culture_filter": "Мәдениет сүзгісі",
        "upload_header": "### 📤 Суретті жүктеу",
        "upload_btn": "🚀 Талдауды бастау",
        "results_header": "### 📊 Талдау нәтижелері",
        "confidence": "Сенімділік",
        "recommendation": "#### 💡 Ұсыныс",
        "prob_dist": "#### Ықтималдықтардың таралуы",
        "class_col": "Класс",
        "prob_col": "Ықтималдық",
        "placeholder": "Фотосуретті жүктеп, 'Талдауды бастау' түймесін басыңыз",
        "loading": "Жапырақ құрылымы талдануда...",
        "success": "Талдау аяқталды!",
        "lang_select": "Тіл таңдау"
    }
}

# Словарь перевода болезней
DISEASE_TRANSLATIONS = {
    "RU": {
        "healthy": "Здоров", "Blight": "Фитофтороз", "Rust": "Ржавчина", "Spot": "Пятнистость",
        "Mold": "Плесень", "Virus": "Вирус", "Mites": "Клещи", "Scab": "Парша", "Rot": "Гниль",
        "Corn": "Кукуруза", "Tomato": "Томат", "Potato": "Картофель", "Pepper": "Перец",
        "Rice": "Рис", "Apple": "Яблоко", "Grape": "Виноград", "Peach": "Персик",
        "Strawberry": "Клубника", "Orange": "Апельсин", "Blueberry": "Черника", "Cherry": "Вишня"
    },
    "KZ": {
        "healthy": "Сау", "Blight": "Фитофтороз", "Rust": "Тот", "Spot": "Дақ",
        "Mold": "Зең", "Virus": "Вирус", "Mites": "Кенелер", "Scab": "Қотыр", "Rot": "Шірік",
        "Corn": "Жүгері", "Tomato": "Қызанақ", "Potato": "Картоп", "Pepper": "Бұрыш",
        "Rice": "Күріш", "Apple": "Алма", "Grape": "Жүзім", "Peach": "Шаттық",
        "Strawberry": "Құлпынай", "Orange": "Апельсин", "Blueberry": "Көкжидек", "Cherry": "Шие"
    }
}

def translate_disease(name, lang):
    if lang == "EN": return name
    
    # Пытаемся перевести по частям
    words = name.replace("_", " ").replace("(", " ").replace(")", " ").split()
    translated_words = []
    
    dict_lang = DISEASE_TRANSLATIONS.get(lang, {})
    
    for word in words:
        found = False
        for key, val in dict_lang.items():
            if key.lower() in word.lower():
                translated_words.append(val)
                found = True
                break
        if not found:
            translated_words.append(word)
            
    return " ".join(translated_words)

# Доп. переводы для DIP
DIP_LANGS = {
    "RU": {
        "tab_ai": "🤖 AI Диагностика",
        "tab_dip": "🔬 DIP Анализ",
        "dip_header": "Методы цифровой обработки изображений",
        "original": "Оригинал",
        "enhanced": "Улучшение контраста (CLAHE)",
        "edges": "Выделение границ (Canny)",
        "mask": "Сегментация (HSV Порог)",
        "dip_desc": "Эти методы используются для предварительной обработки изображений перед подачей в нейросеть.",
        "hist_header": "📊 Технический анализ пикселей",
        "hist_desc": "Гистограмма распределения яркости пикселей (Intensity Histogram). Позволяет оценить освещенность и контрастность снимка.",
        "intensity": "Яркость",
        "count": "Кол-во пикселей",
        "area_header": "📐 Анализ площади (Segmentation Statistics)",
        "area_leaf": "Площадь листа",
        "area_bg": "Фон / Почва"
    },
    "EN": {
        "tab_ai": "🤖 AI Diagnostics",
        "tab_dip": "🔬 DIP Analysis",
        "dip_header": "Digital Image Processing Methods",
        "original": "Original",
        "enhanced": "Contrast Enhancement (CLAHE)",
        "edges": "Edge Detection (Canny)",
        "mask": "Segmentation (HSV Threshold)",
        "dip_desc": "These methods are used for image preprocessing before feeding into the neural network.",
        "hist_header": "📊 Pixel Technical Analysis",
        "hist_desc": "Pixel intensity distribution histogram. Used to evaluate image illumination and contrast.",
        "intensity": "Intensity",
        "count": "Pixel Count",
        "area_header": "📐 Area Analysis (Segmentation Statistics)",
        "area_leaf": "Leaf Area",
        "area_bg": "Background / Soil"
    },
    "KZ": {
        "tab_ai": "🤖 AI Диагностика",
        "tab_dip": "🔬 DIP Талдау",
        "dip_header": "Сандық кескінді өңдеу әдістері",
        "original": "Түпнұсқа",
        "enhanced": "Контрастты жақсарту (CLAHE)",
        "edges": "Шекараларды анықтау (Canny)",
        "mask": "Сегментация (HSV табалдырығы)",
        "dip_desc": "Бұл әдістер нейрондық желіге жіберілгенге дейін кескінді алдын ала өңдеу үшін қолданылады.",
        "hist_header": "📊 Пиксельдерді техникалық талдау",
        "hist_desc": "Пиксель жарықтығының таралу гистограммасы. Кескіннің жарықтануы мен контрастын бағалауға мүмкіндік береді.",
        "intensity": "Жарықтық",
        "count": "Пиксель саны",
        "area_header": "📐 Ауданды талдау (сегменттеу статистикасы)",
        "area_leaf": "Жапырақ ауданы",
        "area_bg": "Фон / Топырақ"
    }
}

import cv2
import numpy as np

def apply_dip(image_pil):
    img = np.array(image_pil)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # 2. Canny
    edges = cv2.Canny(img_gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # 3. HSV Masking
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # 4. Histogram Data
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    
    return enhanced_rgb, edges_rgb, mask_rgb, hist

@st.cache_resource
def load_pipeline():
    return AgroInferencePipeline()

def main():
    # Выбор языка в сайдбаре
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
        st.title("AgroScan AI")
        
        lang = st.radio("Тіл / Язык / Language", ["RU", "KZ", "EN"], horizontal=True)
        t = LANGUAGES[lang]
        dt = DIP_LANGS[lang]
        
        st.markdown("---")
        st.info(t["sidebar_info"])
        
        culture_filter = st.selectbox(
            t["culture_filter"],
            ["auto", "Tomato", "Potato", "Corn", "Pepper", "Rice", "Apple", "Grape", "Orange", "Peach", "Strawberry"],
        )
        
        st.markdown("---")
        st.write("v1.2.0 | DIP Edition")

    # Основной контент
    st.markdown(f"<h1 style='text-align: left;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{t['subtitle']}</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs([dt["tab_ai"], dt["tab_dip"]])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(t["upload_header"])
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="ai_uploader")
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                if st.button(t["upload_btn"]):
                    pipeline = load_pipeline()
                    temp_path = "temp_predict.jpg"
                    image.save(temp_path)
                    
                    with st.spinner(t["loading"]):
                        results = pipeline.run_inference(temp_path, culture=culture_filter)
                    
                    os.remove(temp_path)
                    st.session_state['results'] = results
                    st.success(t["success"])

        with col2:
            st.markdown(t["results_header"])
            if 'results' in st.session_state:
                res = st.session_state['results']
                translated_name = translate_disease(res['raw_name'], lang)
                conf_pct = res['confidence'] * 100
                st.markdown(f"""<div class="prediction-card"><h2 style='margin-top: 0;'>{translated_name}</h2><p style='font-size: 1.2em;'>{t['confidence']}: <b>{conf_pct:.2f}%</b></p></div>""", unsafe_allow_html=True)
                st.progress(res['confidence'])
                st.markdown(t["recommendation"])
                st.warning(res['recommendation'])
                st.markdown(t["prob_dist"])
                all_probs = res['probs']
                class_names = load_pipeline().class_names
                translated_classes = [translate_disease(name, lang) for name in class_names]
                prob_df = pd.DataFrame({t['class_col']: translated_classes, t['prob_col']: all_probs}).sort_values(by=t['prob_col'], ascending=False).head(5)
                fig = px.bar(prob_df, x=t['prob_col'], y=t['class_col'], orientation='h', color=t['prob_col'], color_continuous_scale='Greens', template='plotly_white')
                fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(t["placeholder"])

    with tab2:
        st.markdown(f"### {dt['dip_header']}")
        st.write(dt["dip_desc"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            enhanced, edges, mask, hist = apply_dip(image)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(image, caption=dt["original"], use_container_width=True)
                st.image(enhanced, caption=dt["enhanced"], use_container_width=True)
            with c2:
                st.image(edges, caption=dt["edges"], use_container_width=True)
                st.image(mask, caption=dt["mask"], use_container_width=True)
                
            # 4. Histogram Data
            hist_df = pd.DataFrame({
                dt["intensity"]: list(range(256)),
                dt["count"]: hist
            })
            
            fig_hist = px.line(
                hist_df, x=dt["intensity"], y=dt["count"],
                title=dt["hist_header"], template="plotly_white",
                color_discrete_sequence=['#1b4332']
            )
            fig_hist.update_traces(fill='tozeroy')
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### {dt['area_header']}")
            
            # Расчет площади
            leaf_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            bg_pixels = total_pixels - leaf_pixels
            
            area_df = pd.DataFrame({
                "Category": [dt["area_leaf"], dt["area_bg"]],
                "Pixels": [leaf_pixels, bg_pixels]
            })
            
            fig_pie = px.pie(
                area_df, names="Category", values="Pixels",
                color_discrete_sequence=['#40916c', '#d8f3dc'],
                hole=0.4,
                title=dt["area_header"]
            )
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        else:
            st.info(t["placeholder"])

if __name__ == "__main__":
    main()
