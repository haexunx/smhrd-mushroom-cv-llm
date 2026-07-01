"""느타리버섯 이미지와 환경 센서값을 함께 분석하는 Streamlit 애플리케이션.

전체 처리 흐름:
1. 사용자가 온도·습도·CO2 값과 버섯 이미지를 입력한다.
2. YOLO가 이미지 속 생육 상태 또는 병해를 탐지한다.
3. 탐지 결과를 AI Hub 형태의 JSON 메타데이터로 정리한다.
4. mushroom_guide.md에서 관련 RAG 문맥을 선택한다.
5. CV 결과, 환경값, RAG 문맥을 GPT에 전달해 관리 피드백을 생성한다.
"""

import os
import re
import json
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------------------------
# 기본 경로 및 환경 변수 설정
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).parent

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

GUIDE_PATH = BASE_DIR / "mushroom_guide.md"
MODEL_PATH = BASE_DIR / "models" / "best.pt"


# ----------------------------------------------------------------------
# YOLO 선택적 import
# ----------------------------------------------------------------------

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ----------------------------------------------------------------------
# Streamlit 페이지 설정
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="느타리버섯 생육 진단 & AI 컨설턴트",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 스마트팜 버섯 병해 및 생육 상태 AI 진단 서비스")
st.write(
    "느타리버섯 이미지 분석 결과와 실시간 환경값을 바탕으로, "
    "RAG 문서(mushroom_guide.md)를 참고해 AI 관리 피드백을 제공합니다."
)


# ----------------------------------------------------------------------
# YOLO 클래스 및 AI Hub 메타데이터 매핑
# ----------------------------------------------------------------------

YOLO_TO_AIHUB_MAP = {
    # 영어 클래스명
    "normal": {"korean_name": "정상", "is_normal": True},
    "dry": {"korean_name": "생육불량 (건조)", "is_normal": False},
    "over_humidity": {"korean_name": "생육불량 (과습)", "is_normal": False},
    "co2_high": {"korean_name": "생육불량 (환기부족/CO2 과다)", "is_normal": False},
    "harvest_ready": {"korean_name": "수확적기", "is_normal": True},
    "bacterial_brown_blotch": {"korean_name": "세균갈색무늬병", "is_normal": False},
    "bacterial_browning": {"korean_name": "세균성갈변병", "is_normal": False},
    "blue_mold": {"korean_name": "푸른곰팡이병", "is_normal": False},
    "green_mold": {"korean_name": "푸른곰팡이병", "is_normal": False},
    "white_mold": {"korean_name": "흰곰팡이", "is_normal": False},
    "cobweb_mold": {"korean_name": "솜털곰팡이", "is_normal": False},
    "black_rot": {"korean_name": "세균성검은썩음병", "is_normal": False},
    "other_disease": {"korean_name": "기타 병해", "is_normal": False},

    # 숫자 클래스명
    "0": {"korean_name": "정상", "is_normal": True},
    "1": {"korean_name": "생육불량 (건조)", "is_normal": False},
    "2": {"korean_name": "생육불량 (과습)", "is_normal": False},
    "3": {"korean_name": "생육불량 (환기부족/CO2 과다)", "is_normal": False},
    "4": {"korean_name": "수확적기", "is_normal": True},
    "5": {"korean_name": "세균갈색무늬병", "is_normal": False},
    "6": {"korean_name": "푸른곰팡이병", "is_normal": False},
    "7": {"korean_name": "기타 병해", "is_normal": False},
}

BASE_CV_OUTPUT = {
    "IMAGE": {
        "IMAGE_FILE_NAME": "",
        "ANNOTATION_COUNT": 0
    },
    "ANNOTATION_INFO": [],
    "META": {
        "DBYHS_SPCHCKN": "정상",
        "DBYHS_NORMALITY_ALTERNATIVE": True,
        "TEMPERATURE": 15.0,
        "HUMIDITY": 90.0,
        "CARBON_DIOXIDE": 1000.0
    }
}


# ----------------------------------------------------------------------
# YOLO 모델 로드
# ----------------------------------------------------------------------

@st.cache_resource
def load_yolo_model(model_path=MODEL_PATH):
    if not YOLO_AVAILABLE:
        return None

    if not model_path.exists():
        return None

    try:
        return YOLO(str(model_path))
    except Exception as e:
        st.warning(f"YOLO 모델 로드 실패: {e}")
        return None


yolo_model = load_yolo_model()


# ----------------------------------------------------------------------
# RAG 문서 처리 함수
# ----------------------------------------------------------------------

def load_guide_text():
    if not GUIDE_PATH.exists():
        return ""

    try:
        return GUIDE_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        st.warning(f"mushroom_guide.md 로드 중 에러: {e}")
        return ""


def split_markdown_sections(markdown_text):
    """mushroom_guide.md를 ## 제목 기준으로 분리한다."""
    sections = {}
    pattern = r"^##\s+(.+)$"
    matches = list(re.finditer(pattern, markdown_text, flags=re.MULTILINE))

    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections[title] = content

    return sections


def add_section_if_match(sections, selected, keyword):
    for title, content in sections.items():
        if keyword in title:
            selected.append((title, content))


def retrieve_rag_context(guide_text, meta):
    """CV 결과와 환경값에 맞는 RAG 문맥만 선택한다."""
    if not guide_text:
        return ""

    sections = split_markdown_sections(guide_text)

    if not sections:
        return guide_text[:6000]

    disease_name = str(meta.get("DBYHS_SPCHCKN", "정상"))
    is_normal = meta.get("DBYHS_NORMALITY_ALTERNATIVE", True)
    humidity = meta.get("HUMIDITY")
    co2 = meta.get("CARBON_DIOXIDE")

    selected = []

    # 1. CV 결과 기반 섹션
    if is_normal:
        if "수확" in disease_name:
            add_section_if_match(sections, selected, "harvest_ready")
        else:
            add_section_if_match(sections, selected, "normal")
    else:
        if "건조" in disease_name:
            add_section_if_match(sections, selected, "dry")
        elif "과습" in disease_name:
            add_section_if_match(sections, selected, "over_humidity")
        elif "CO2" in disease_name or "환기" in disease_name or "이산화탄소" in disease_name:
            add_section_if_match(sections, selected, "co2_high")
        elif "푸른곰팡이" in disease_name:
            add_section_if_match(sections, selected, "푸른곰팡이병")
        elif "세균갈색" in disease_name or "세균성갈변" in disease_name or "갈변" in disease_name:
            add_section_if_match(sections, selected, "세균갈색무늬병")
            add_section_if_match(sections, selected, "세균성갈변병")
        elif "흰곰팡이" in disease_name:
            add_section_if_match(sections, selected, "흰곰팡이")
        elif "솜털곰팡이" in disease_name:
            add_section_if_match(sections, selected, "솜털곰팡이")
        elif "검은썩음" in disease_name:
            add_section_if_match(sections, selected, "세균성검은썩음병")

    # 2. 환경값 기반 섹션
    if humidity is not None and humidity < 70:
        add_section_if_match(sections, selected, "dry")
        add_section_if_match(sections, selected, "습도 및 배지 수분 관리")

    if humidity is not None and humidity > 90:
        add_section_if_match(sections, selected, "over_humidity")
        add_section_if_match(sections, selected, "습도 및 배지 수분 관리")

    if co2 is not None and co2 > 1500:
        add_section_if_match(sections, selected, "co2_high")
        add_section_if_match(sections, selected, "환기 관리")

    # 3. CV 결과와 환경값 불일치 또는 normal이지만 환경 위험값이 있는 경우
    if (
        (is_normal and humidity is not None and (humidity < 70 or humidity > 90))
        or (is_normal and co2 is not None and co2 > 1500)
        or ("건조" in disease_name and humidity is not None and humidity > 90)
        or ("과습" in disease_name and humidity is not None and humidity < 70)
    ):
        add_section_if_match(sections, selected, "CV 결과와 환경값 불일치 해석")

    # 4. 병해가 있는 경우 공통 방제/위생 섹션
    if not is_normal and disease_name not in ["정상", "수확적기"]:
        add_section_if_match(sections, selected, "재배 위생 관리")
        add_section_if_match(sections, selected, "병해충 방제 공통 주의사항")

    # 5. 체크리스트는 항상 추가
    add_section_if_match(sections, selected, "재배 환경 점검 체크리스트")

    # 6. 아무것도 선택되지 않으면 normal 섹션
    if not selected:
        add_section_if_match(sections, selected, "normal")

    # 7. 중복 제거
    unique_sections = {}
    for title, content in selected:
        unique_sections[title] = content

    context = ""
    for title, content in unique_sections.items():
        context += f"\n[문서 섹션: {title}]\n{content}\n"

    return context.strip()


# ----------------------------------------------------------------------
# UI 구성
# ----------------------------------------------------------------------

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🎛️ 1. 스마트팜 환경 센서 설정")
    st.write("진단 시뮬레이션을 위해 현재 버섯 재배 하우스의 환경 값을 설정하세요.")

    edited_temp = st.slider(
        "실내 온도 (°C)",
        min_value=0.0,
        max_value=45.0,
        value=15.0,
        step=0.1
    )

    edited_humid = st.slider(
        "실내 습도 (%)",
        min_value=1.0,
        max_value=100.0,
        value=90.0,
        step=0.1
    )

    edited_co2 = st.slider(
        "이산화탄소 농도 (CO2, ppm)",
        min_value=300,
        max_value=10000,
        value=1000,
        step=10
    )

    st.markdown("---")
    st.subheader("📸 2. 버섯 이미지 업로드 및 분석")

    uploaded_file = st.file_uploader(
        "버섯 이미지를 업로드해주세요 (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        cv_data = json.loads(json.dumps(BASE_CV_OUTPUT))
        cv_data["IMAGE"]["IMAGE_FILE_NAME"] = uploaded_file.name
        cv_data["META"]["TEMPERATURE"] = edited_temp
        cv_data["META"]["HUMIDITY"] = edited_humid
        cv_data["META"]["CARBON_DIOXIDE"] = edited_co2

        st.markdown("---")
        st.markdown("### 🔍 YOLO 분석 결과")

        detected_labels = []
        annotations = []

        if yolo_model is not None:
            with st.spinner("YOLO 모델 분석 중..."):
                results = yolo_model(image)

                annotated_img_array = results[0].plot()
                annotated_image = Image.fromarray(annotated_img_array[..., ::-1])
                st.image(annotated_image, caption="YOLO 탐지 영역 시각화", use_container_width=True)

                boxes = results[0].boxes
                names = yolo_model.names

                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    raw_label = names[cls_id]

                    xywh = box.xywh[0].tolist()
                    annotations.append({
                        "ID": idx + 1,
                        "BOUNDING_BOX_X_COORDINATE": int(xywh[0]),
                        "BOUNDING_BOX_Y_COORDINATE": int(xywh[1]),
                        "BOUNDING_BOX_WIDTH": int(xywh[2]),
                        "BOUNDING_BOX_HEIGHT": int(xywh[3])
                    })

                    mapped_info = YOLO_TO_AIHUB_MAP.get(str(raw_label)) or YOLO_TO_AIHUB_MAP.get(str(cls_id))

                    if mapped_info:
                        korean_name = mapped_info["korean_name"]
                        is_normal = mapped_info["is_normal"]
                    else:
                        korean_name = str(raw_label)
                        is_normal = True if "normal" in str(raw_label).lower() else False

                    detected_labels.append({
                        "name": korean_name,
                        "confidence": f"{conf:.2%}",
                        "is_normal": is_normal
                    })

            cv_data["ANNOTATION_INFO"] = annotations
            cv_data["IMAGE"]["ANNOTATION_COUNT"] = len(annotations)

            disease_candidates = [item for item in detected_labels if not item["is_normal"]]

            if disease_candidates:
                cv_data["META"]["DBYHS_SPCHCKN"] = disease_candidates[0]["name"]
                cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = False
            else:
                normal_candidates = [item for item in detected_labels if item["is_normal"]]
                cv_data["META"]["DBYHS_SPCHCKN"] = normal_candidates[0]["name"] if normal_candidates else "정상"
                cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = True

        else:
            st.info("💡 YOLO 모델이 없어서 데모용 CV 결과를 사용합니다.")
            st.image(image, caption="업로드된 이미지 (데모 분석)", use_container_width=True)

            demo_class = st.selectbox(
                "데모용 CV 결과 선택",
                [
                    "정상",
                    "생육불량 (건조)",
                    "생육불량 (과습)",
                    "생육불량 (환기부족/CO2 과다)",
                    "수확적기",
                    "푸른곰팡이병",
                    "세균갈색무늬병",
                    "세균성갈변병",
                    "흰곰팡이",
                    "솜털곰팡이",
                    "세균성검은썩음병"
                ]
            )

            demo_is_normal = demo_class in ["정상", "수확적기"]

            detected_labels = [
                {
                    "name": demo_class,
                    "confidence": "94.5%",
                    "is_normal": demo_is_normal
                }
            ]

            cv_data["META"]["DBYHS_SPCHCKN"] = demo_class
            cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = demo_is_normal
            cv_data["IMAGE"]["ANNOTATION_COUNT"] = 1
            cv_data["ANNOTATION_INFO"] = [{
                "ID": 1,
                "BOUNDING_BOX_X_COORDINATE": 200,
                "BOUNDING_BOX_Y_COORDINATE": 1200,
                "BOUNDING_BOX_WIDTH": 300,
                "BOUNDING_BOX_HEIGHT": 300
            }]

        st.write("**실시간 탐지 라벨:**")
        for item in detected_labels:
            st.write(f"- 🔍 **{item['name']}** (신뢰도: {item['confidence']})")

        with st.expander("🔍 연동 중인 최종 JSON 메타데이터 확인"):
            st.json(cv_data)

    else:
        cv_data = None


with col2:
    st.subheader("📊 3. AI 환경 진단 및 관리 피드백")

    if uploaded_file is not None and cv_data is not None:
        meta = cv_data.get("META", {})

        st.info("**🍄 분석 대상 품종:** 느타리버섯 (Oyster Mushroom)")

        with st.expander("📖 느타리버섯 생육 환경 참고표"):
            st.markdown(
                "| 항목 | 참고 내용 |\n"
                "| :--- | :--- |\n"
                "| 온도 | 생육 단계와 재배 방식에 따라 달라질 수 있으므로 단독으로 정상 여부를 단정하지 않음 |\n"
                "| 습도 | 낮으면 건조 가능성, 높으면 과습 및 병해 위험 가능성을 함께 점검 |\n"
                "| CO2 | 높으면 환기 부족 또는 공기 정체 가능성 점검 |\n"
                "| 주의 | CV 결과와 환경값이 충돌하면 센서 위치, 촬영 시점, 배지 수분 상태를 함께 확인 |\n"
            )

        is_normal = meta.get("DBYHS_NORMALITY_ALTERNATIVE", True)
        disease_name = meta.get("DBYHS_SPCHCKN", "정상")

        if is_normal:
            st.success(f"✅ **CV 결과:** 이미지상 정상 또는 수확 가능 상태로 판단됨 ({disease_name})")
        else:
            st.error(f"🚨 **CV 결과:** 생육불량 또는 병해 의심 ({disease_name})")

        m1, m2, m3 = st.columns(3)
        m1.metric("실내 온도", f"{meta.get('TEMPERATURE')} °C")
        m2.metric("실내 습도", f"{meta.get('HUMIDITY')} %")
        m3.metric("CO2 농도", f"{meta.get('CARBON_DIOXIDE')} ppm")

        if st.button("🤖 AI 관리 피드백 생성"):
            openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

            if not openai_api_key:
                st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
            else:
                with st.spinner("RAG 문맥 기반 AI 피드백 생성 중..."):
                    try:
                        guide_text = load_guide_text()

                        if not guide_text:
                            st.error("mushroom_guide.md 파일을 찾을 수 없거나 내용이 비어 있습니다.")
                            st.stop()

                        retrieved_context = retrieve_rag_context(
                            guide_text=guide_text,
                            meta=meta
                        )

                        if not retrieved_context:
                            st.error("검색된 RAG 문맥이 없습니다.")
                            st.stop()

                        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

                        llm = ChatOpenAI(
                            model=model_name,
                            temperature=0.2,
                            openai_api_key=openai_api_key
                        )

                        prompt_template = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                (
                                    "당신은 느타리버섯 스마트팜 생육 관리 보조 AI입니다. "
                                    "CV 분석 결과, 재배 환경값, 검색된 RAG 문맥을 바탕으로 "
                                    "재배자가 이해하기 쉬운 원인 설명과 관리 조치 추천을 제공합니다. "
                                    "반드시 검색된 RAG 문맥에 근거해서 답변하세요. "
                                    "RAG 문맥에 없는 내용은 단정하지 마세요. "
                                    "CV 결과는 보조 판단이며 최종 진단이 아닙니다. "
                                    "약제 사용, 식용 가능 여부, 폐기 여부는 반드시 전문가 확인이 필요하다고 안내하세요."
                                )
                            ),
                            (
                                "user",
                                (
                                    "[CV 분석 결과]\n"
                                    "상태 또는 병해 라벨: {disease}\n"
                                    "정상 여부: {is_normal}\n\n"

                                    "[환경 입력값]\n"
                                    "온도: {temp}도\n"
                                    "습도: {humidity}%\n"
                                    "CO2: {co2}ppm\n\n"

                                    "[검색된 RAG 문맥]\n"
                                    "{retrieved_context}\n\n"

                                    "위 정보를 바탕으로 아래 형식으로 답변하세요.\n\n"

                                    "1. 현재 상태 요약\n"
                                    "- CV 결과와 환경값을 함께 고려해 현재 상태를 요약하세요.\n"
                                    "- CV 결과가 정상이어도 환경값에 위험 신호가 있으면 정상으로 단정하지 말고, "
                                    "'이미지상 정상으로 보이나 환경 관리 점검이 필요함'으로 표현하세요.\n"
                                    "- 특히 CO2가 높을 때는 무조건 문제라고 하지 말고, 현재 생육 단계에서 의도적으로 유지 중인 값인지 확인이 필요하다고 설명하세요.\n\n"

                                    "2. 의심 원인\n"
                                    "- 검색된 RAG 문맥에 있는 원인을 중심으로 설명하세요.\n"
                                    "- CV 결과와 환경값이 충돌하면 센서 위치, 배지 수분 상태, 촬영 시점 차이, CV 오분류 가능성을 함께 언급하세요.\n\n"

                                    "3. 환경값 해석\n"
                                    "- 온도, 습도, CO2 각각을 해석하세요.\n"
                                    "- RAG 문맥에 명확한 기준값이 없는 경우 '정상 범위', '적정 범위', '안전한 수치'라고 단정하지 마세요.\n"
                                    "- 특히 온도에 대해서는 생육 단계가 명확하지 않으면 '적정 범위, '정상범위'라고 말하지 말고 입력값으로만 설명하세요./n"
                                    "- 기준값이 없는 환경값은 '현재 입력값 기준으로 관찰 필요', '주요 원인으로 단정하기 어려움', '추가 확인 필요'처럼 표현하세요.\n"
                                    "- CO2가 높으면 환기 부족 가능성을 설명하되, 환기 조절 후 습도 저하가 생길 수 있음을 함께 언급하세요.\n\n"
                                    "- CO2가 높게 입력된 경우에도 무조건 환기 부족이나 이상 상태로 단정하지 마세요. 현재 생육 단계나 재배 목표에 따라 의도적으로 유지 중인 값일 수 있으므로, '생육 단계 확인 필요'라고 표현하세요.\n"
                                    "- 단, 자실체 생육기라면 높은 CO2가 환기 부족, 대 길어짐, 갓 발달 저하와 관련될 수 있으므로 자실체 형태와 환기 상태를 함께 점검하라고 안내하세요.\n"
                                    "4. 조치 추천\n"
                                    "- 재배자가 바로 확인하거나 조절할 수 있는 행동을 제안하세요.\n"
                                    "- 환기를 조절할 때는 습도 저하 가능성을 함께 고려하세요.\n"
                                    "- 습도를 조절할 때는 과도한 분무로 인한 과습과 곰팡이성 병해 위험을 함께 고려하세요.\n"
                                    "- 예: 배지 수분 확인, 자실체 표면 확인, 센서 위치 확인, 환기량 점검, 분무량 점검, 오염 개체 격리 등\n\n"

                                    "5. 주의사항\n"
                                    "- CV 결과는 보조 판단이며 최종 진단이 아니라고 안내하세요.\n"
                                    "- 약제 사용, 식용 가능 여부, 폐기 여부는 전문가 확인이 필요하다고 안내하세요.\n"
                                    "- 과도한 분무와 급격한 환기 조절의 위험을 함께 안내하세요.\n\n"

                                    "6. 다음 확인 항목\n"
                                    "- 다음 관찰 시 확인해야 할 항목을 체크리스트로 작성하세요.\n"
                                    "- 배지 수분 상태, 자실체 표면 상태, 환기 후 습도 변화, CO2 변화 추이, 센서 위치를 우선 포함하세요.\n\n"

                                    "답변은 발표 데모에 적합하게 간결하고 실무적으로 작성하세요."
                                )
                            )
                        ])

                        chain = prompt_template | llm | StrOutputParser()

                        feedback = chain.invoke({
                            "disease": disease_name,
                            "is_normal": is_normal,
                            "temp": meta.get("TEMPERATURE"),
                            "humidity": meta.get("HUMIDITY"),
                            "co2": meta.get("CARBON_DIOXIDE"),
                            "retrieved_context": retrieved_context
                        })

                        st.markdown("---")
                        st.markdown("### 📋 AI 관리 피드백")

                        with st.expander("검색된 RAG 문맥 확인"):
                            st.text(retrieved_context)

                        st.markdown(feedback)

                    except Exception as e:
                        st.error(f"AI 피드백 생성 중 에러가 발생했습니다: {e}")

    else:
        st.info("👈 왼쪽 영역에서 이미지를 먼저 업로드해 주세요.")