import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import json

# Настройки страницы
st.set_page_config(
    page_title="Satellite Building Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🏢 Satellite Building Area Calculator</h1>', unsafe_allow_html=True)

# Сайдбар
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка модели
    st.subheader("Модель")
    model_option = st.selectbox(
        "Выберите модель",
        ["DeepLabV3+ (ResNet50)", "U-Net (ResNet34)", "PSPNet (ResNet50)"],
        index=0
    )
    
    st.subheader("Параметры сегментации")
    confidence_threshold = st.slider(
        "Порог уверенности", 
        0.1, 0.9, 0.5, 0.05,
        help="Порог для бинаризации маски"
    )
    
    st.subheader("Масштабирование")
    gsd_option = st.radio(
        "Метод определения масштаба",
        ["Автоматически (детекция объектов)", "Вручную", "Из EXIF данных"]
    )
    
    if gsd_option == "Вручную":
        gsd_value = st.number_input(
            "GSD (м/пиксель)", 
            min_value=0.01, 
            max_value=10.0, 
            value=0.3, 
            step=0.01,
            help="Ground Sampling Distance - метров на пиксель"
        )
    else:
        gsd_value = None
    
    st.subheader("Дополнительные опции")
    show_heatmap = st.checkbox("Показать тепловую карту уверенности", True)
    save_results = st.checkbox("Сохранить результаты", False)
    
    st.divider()
    
    # Информация
    st.info("""
    **Инструкция:**
    1. Загрузите спутниковое изображение
    2. Настройте параметры
    3. Нажмите "Анализировать"
    4. Просмотрите результаты
    """)

# Основная область
col1, col2 = st.columns([2, 1])

with col1:
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "📤 Загрузите спутниковое изображение",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'],
        help="Поддерживаются форматы: JPG, PNG, TIFF"
    )
    
    if uploaded_file is not None:
        # Сохранение временного файла
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Отображение изображения
        image = Image.open(tmp_path)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        # Информация об изображении
        st.caption(f"Размер: {image.size[0]} × {image.size[1]} пикселей")
        
        # Кнопка анализа
        if st.button("🚀 Анализировать изображение", type="primary"):
            with st.spinner("Выполняется анализ..."):
                # Здесь будет вызов вашей модели
                # Покажем примерные результаты для демонстрации
                
                progress_bar = st.progress(0)
                
                # Имитация работы
                for percent_complete in range(100):
                    # Ваш реальный код здесь
                    # segmenter.calculate_building_area(...)
                    progress_bar.progress(percent_complete + 1)
                
                # Примерные результаты
                results = {
                    'total_area_m2': 12560.45,
                    'building_area_m2': 3456.78,
                    'building_percentage': 27.5,
                    'num_buildings': 42,
                    'avg_building_size_m2': 82.3,
                    'coverage_density': 0.45,
                    'gsd_estimated': 0.32,
                    'confidence_score': 0.87
                }
                
                # Сохраняем результаты
                if save_results:
                    with open('results.json', 'w') as f:
                        json.dump(results, f)
                
                st.success("Анализ завершен!")

with col2:
    # Панель результатов
    st.header("📊 Результаты")
    
    if uploaded_file is not None and 'results' in locals():
        # Метрики в карточках
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                "Общая площадь", 
                f"{results['total_area_m2']:,.0f} м²",
                help="Площадь анализируемой территории"
            )
        
        with col_b:
            st.metric(
                "Площадь застройки", 
                f"{results['building_area_m2']:,.0f} м²",
                f"{results['building_percentage']:.1f}%",
                help="Площадь, занятая зданиями"
            )
        
        # Дополнительная информация
        with st.expander("📈 Детальная статистика", expanded=True):
            st.metric("Количество зданий", f"{results['num_buildings']}")
            st.metric("Средний размер здания", f"{results['avg_building_size_m2']:.1f} м²")
            st.metric("Плотность застройки", f"{results['coverage_density']:.2f}")
            st.metric("Расчетный GSD", f"{results['gsd_estimated']:.3f} м/px")
            st.metric("Достоверность", f"{results['confidence_score']:.2%}")
        
        # Визуализация
        with st.expander("👁️ Визуализация"):
            # Создаем примерную визуализацию
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Пример данных
            categories = ['Жилые', 'Коммерческие', 'Промышленные', 'Другие']
            areas = [1200, 800, 1000, 456.78]
            
            axes[0].pie(areas, labels=categories, autopct='%1.1f%%')
            axes[0].set_title('Распределение типов зданий')
            
            # Гистограмма размеров
            sizes = np.random.normal(80, 30, 42)
            axes[1].hist(sizes, bins=15, alpha=0.7, color='skyblue')
            axes[1].set_xlabel('Размер здания (м²)')
            axes[1].set_ylabel('Количество')
            axes[1].set_title('Распределение размеров зданий')
            axes[1].axvline(x=np.mean(sizes), color='red', linestyle='--', 
                          label=f'Среднее: {np.mean(sizes):.1f} м²')
            axes[1].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Кнопки экспорта
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # JSON
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Скачать JSON",
                data=json_str,
                file_name="building_analysis.json",
                mime="application/json"
            )
        
        with col_export2:
            # CSV
            df = pd.DataFrame([results])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name="building_analysis.csv",
                mime="text/csv"
            )
        
        # Предупреждения
        if results['confidence_score'] < 0.7:
            st.warning("⚠️ Низкая достоверность результатов. Проверьте качество изображения.")
        
        if results['building_percentage'] > 50:
            st.error("🚨 Высокая плотность застройки! Превышены нормативы.")
        
    else:
        st.info("👈 Загрузите изображение и нажмите 'Анализировать' для получения результатов")
        
        # Примеры изображений
        with st.expander("📚 Примеры изображений"):
            st.write("Рекомендуемые характеристики:")
            st.markdown("""
            - **Разрешение:** 0.1-1.0 м/пиксель
            - **Формат:** RGB, 8-16 бит
            - **Размер:** 1000×1000 - 5000×5000 пикселей
            - **Облачность:** < 20%
            - **Угол съемки:** надир ±15°
            """)

# Нижняя панель
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🔄 Версия модели: 1.0.0")

with footer_col2:
    st.caption("📅 Последнее обновление: 2024")

with footer_col3:
    st.caption("🔧 Технологии: PyTorch, Streamlit, OpenCV")
