import streamlit as st
import shutil
import cv2
from pathlib import Path
from os.path import basename
from elements.FrameElement import FrameElement
from streamlit_image_zoom import image_zoom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
import yaml

from core import AIHelper


with open("config.yaml", encoding="utf-8") as stream:
    CONFIG = yaml.safe_load(stream)

TEMP_FOLDER_PTH = "temp"
ZOOM_FACTOR = 3

CLEAN_LABEL = CONFIG["common"]["clean_label"]
DIRTY_LABEL = CONFIG["common"]["dirty_label"]

PREDICTION_GOOD = "Правильно"
PREDICTION_BAD = "Ошибка"

ai_helper = AIHelper(CONFIG)


def set_page_static_info():
    st.set_page_config(
        page_title="Норникель. Грязные дела",
        page_icon="configs/logo.jpg",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Норникель-хаккатон. Кейс 'грязные дела'")


def upload_files():
    uploaded_files = st.sidebar.file_uploader(
        "Выберите изображение с камер",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
    )

    if len(uploaded_files) > 0:
        shutil.rmtree(TEMP_FOLDER_PTH, True)
        Path(TEMP_FOLDER_PTH).mkdir(parents=True, exist_ok=True)

        file_saved_paths = []
        for uploaded_file in uploaded_files:
            file_saved_path = f"{TEMP_FOLDER_PTH}/{uploaded_file.name}"
            file_saved_paths.append(file_saved_path)
            with open(file_saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        return file_saved_paths
    else:
        return None


def inference_AI(img_path: str):
    with st.spinner("Запрос обрабатывается, ждите..."):
        frame_element = ai_helper.predict(img_path)

    return frame_element


def show_row_images(frame_element: FrameElement, img_size=500):
    img = frame_element.image
    mask = frame_element.mask_over_img

    col_first, col_second = st.columns(2)
    with col_first:
        st.write("Исходное фото:")
        image_zoom(img, "both", img_size, True, True, ZOOM_FACTOR)
    with col_second:
        st.write("Маска загрязнений (отмечена красным):")
        image_zoom(mask, "both", img_size, True, True, ZOOM_FACTOR)


def draw_results_for_frame(frame_element: FrameElement):
    color = "green" if frame_element.cls == CLEAN_LABEL else "red"
    print("frame_element.cls", frame_element.cls, "CLEAN_LABEL", CLEAN_LABEL, color)
    st.markdown(f"### {frame_element.filename} - :{color}[{frame_element.cls}]")

    show_row_images(frame_element)
    st.dataframe(frame_element.characteristics, hide_index=True)

    st.radio(
        "Оценка качества",
        [
            f":green[{PREDICTION_GOOD}]",
            f":red[{PREDICTION_BAD}]",
        ],
        key=f"human_quality_{frame_element.filename}",
    )

    st.markdown("---")


def get_stats_table(frame_elements: list[FrameElement]):
    df_stats = {"Имя файла": [], "Оценка человека": [], "Предикт ИИ": []}
    for frame_element in frame_elements:
        df_stats["Имя файла"].append(frame_element.filename)
        df_stats["Предикт ИИ"].append(frame_element.cls)

        human_quality = st.session_state[f"human_quality_{frame_element.filename}"]
        human_quality = PREDICTION_GOOD if PREDICTION_GOOD in human_quality else PREDICTION_BAD
        df_stats["Оценка человека"].append(human_quality)

        frame_characteristics = frame_element.characteristics
        for _, frame_characteristic in frame_characteristics.iterrows():
            key = frame_characteristic["Характеристика"]
            value = frame_characteristic["Значение"]
            if key not in df_stats:
                df_stats[key] = []

            df_stats[key].append(value)

    return pd.DataFrame(df_stats)


def main():
    set_page_static_info()
    file_saved_paths = upload_files()

    # 1. Подгрузка
    if file_saved_paths:
        with st.form("Processing"):
            submitted = st.form_submit_button(":green[Получить результаты]")
            if submitted:
                st.session_state["show_result_btn_pressed"] = True
    else:
        st.warning("Пожалуйста, загрузите изображения для обработки.")
        # очистка переменной состояния (аналогия перезапуска страницы):
        st.session_state.clear()

    # 2. Отрисовка вывода результатов + ввод номеров ящиков + подготовка к скачиванию
    frame_elements = []
    if "show_result_btn_pressed" in st.session_state:
        st.write("## Результаты:")
        with st.form("Aggregation"):
            for file_saved_path in file_saved_paths:
                frame_element = inference_AI(file_saved_path)
                draw_results_for_frame(frame_element)
                frame_elements.append(frame_element)
            form_table = st.form_submit_button(":green[Анализировать статистику]")
            if form_table:
                st.session_state["form_table"] = True

    if "form_table" in st.session_state:
        st.markdown(f"## Итоговая статистика:")
        df_stats = get_stats_table(frame_elements)

        # st.pyplot(df_stats, x="Оценка человека", stack=False)
        df_stat_values = df_stats.drop(columns=["Имя файла"])
        ncols = len(df_stat_values.columns)
        fig, axes = plt.subplots(ncols=ncols, figsize=(3 * ncols, 4))

        for col, ax in zip(df_stat_values.columns, axes):
            values = df_stat_values[col]

            if isinstance(values[0], numbers.Number):
                df_stats.hist(col, ax=ax)
            else:
                label_counts = values.value_counts()
                values = label_counts.values
                labels = [label.replace(" ", "\n") for label in label_counts.index]
                ax.bar(labels, values)

                ax.set_title(col)
                ax.set_ylabel("количество кадров")
                ax.grid()

        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.dataframe(df_stats, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    # Загрузка конфигурации программы:
    main()
