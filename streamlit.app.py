import streamlit as st
import os
import json
from PIL import Image
from dotenv import load_dotenv

# dotenv 로드 (.env 파일에서 API 키 읽기, 터미널 세션 캐시 덮어쓰기 강제)
load_dotenv(override=True)

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# YOLO 라이브러리 임포트 (ultralytics 설치 필요)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 페이지 레이아웃 및 스타일 설정
st.set_page_config(
    page_title="느타리버섯 생육 진단 & AI 컨설턴트",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 스마트팜 버섯 병해 및 생육 상태 AI 진단 서비스")
st.write("버섯 이미지 분석에서 추출된 AI Hub 표준 메타데이터와 실시간 환경 제어 변수들을 바탕으로 AI(gpt-4o-mini)가 정밀 피드백을 제공합니다.")

# ----------------------------------------------------------------------
# 📌 YOLO 클래스 및 AI Hub 메타데이터 매핑 정보 정의
# ----------------------------------------------------------------------
# 학습한 YOLO 모델의 아웃풋 클래스명(또는 인덱스 문자열)을 한국어 질병명 및 정상 여부로 변환합니다.
YOLO_TO_AIHUB_MAP = {
    # 1. 영어 문자열 라벨 매핑 (실제 모델 학습 클래스명 매핑)
    "normal": {"korean_name": "정상", "is_normal": True},
    "dry": {"korean_name": "생육불량 (건조)", "is_normal": False},
    "over_humidity": {"korean_name": "생육불량 (과습)", "is_normal": False},
    "co2_high": {"korean_name": "생육불량 (환기부족/CO2 과다)", "is_normal": False},
    "harvest_ready": {"korean_name": "수확적기", "is_normal": True},
    "bacterial_brown_blotch": {"korean_name": "세균갈색무늬병", "is_normal": False},
    "blue_mold": {"korean_name": "푸른곰팡이병", "is_normal": False},
    "other_disease": {"korean_name": "기타 병해", "is_normal": False},
    
    # 2. 인덱스 번호(문자열) 라벨 매핑
    "0": {"korean_name": "정상", "is_normal": True},
    "1": {"korean_name": "생육불량 (건조)", "is_normal": False},
    "2": {"korean_name": "생육불량 (과습)", "is_normal": False},
    "3": {"korean_name": "생육불량 (환기부족/CO2 과다)", "is_normal": False},
    "4": {"korean_name": "수확적기", "is_normal": True},
    "5": {"korean_name": "세균갈색무늬병", "is_normal": False},
    "6": {"korean_name": "푸른곰팡이병", "is_normal": False},
    "7": {"korean_name": "기타 병해", "is_normal": False}
}

# 기본 메타데이터 구조 템플릿 (하드코딩된 상수들을 제외하고 동적 데이터만 수록)
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
# 📌 YOLO 모델 캐싱 로드
# ----------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(model_path="models/best.pt"):
    if not YOLO_AVAILABLE:
        return None
    try:
        if os.path.exists(model_path):
            return YOLO(model_path)
    except Exception as e:
        st.warning(f"YOLO 모델 로드 실패: {e}")
    return None

yolo_model = load_yolo_model()

# ----------------------------------------------------------------------
# 📌 UI 구성
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🎛️ 1. 스마트팜 환경 센서 설정")
    st.write("진단 시뮬레이션을 위해 현재 버섯 재배 하우스의 환경 값을 설정하세요.")
    
    # 온도, 습도, 이산화탄소 농도 슬라이더 (항상 표시)
    edited_temp = st.slider("실내 온도 (°C)", min_value=5.0, max_value=35.0, value=15.0, step=0.1)
    edited_humid = st.slider("실내 습도 (%)", min_value=50.0, max_value=100.0, value=90.0, step=0.1)
    edited_co2 = st.slider("이산화탄소 농도 (CO2, ppm)", min_value=300, max_value=3000, value=1000, step=10)
    
    st.markdown("---")
    st.subheader("📸 2. 버섯 이미지 업로드 및 분석")
    
    # 이미지 업로드 (항상 표시)
    uploaded_file = st.file_uploader("버섯 이미지를 업로드해주세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # 기본 템플릿 복사 후 입력값 바인딩
        cv_data = json.loads(json.dumps(BASE_CV_OUTPUT)) # deep copy
        cv_data["IMAGE"]["IMAGE_FILE_NAME"] = uploaded_file.name
        cv_data["META"]["TEMPERATURE"] = edited_temp
        cv_data["META"]["HUMIDITY"] = edited_humid
        cv_data["META"]["CARBON_DIOXIDE"] = edited_co2
        
        # ----------------------------------------------------------------------
        # 📌 YOLO 이미지 분석 및 메타데이터 추출 실행
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔍 YOLO 분석 결과")
        
        detected_labels = []
        is_normal_status = True
        annotations = []
        
        if yolo_model is not None:
            with st.spinner("YOLO 모델 분석 중..."):
                results = yolo_model(image)
                
                # 예측 이미지 시각화
                annotated_img_array = results[0].plot()
                annotated_image = Image.fromarray(annotated_img_array[..., ::-1]) # BGR to RGB
                st.image(annotated_image, caption="YOLO 탐지 영역 시각화", width="stretch")
                
                boxes = results[0].boxes
                names = yolo_model.names
                
                for idx, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    raw_label = names[cls_id]
                    
                    # 바운딩 박스 좌표 추출 (x, y, w, h)
                    xywh = box.xywh[0].tolist()
                    annotations.append({
                        "ID": idx + 1,
                        "BOUNDING_BOX_X_COORDINATE": int(xywh[0]),
                        "BOUNDING_BOX_Y_COORDINATE": int(xywh[1]),
                        "BOUNDING_BOX_WIDTH": int(xywh[2]),
                        "BOUNDING_BOX_HEIGHT": int(xywh[3])
                    })
                    
                    # YOLO_TO_AIHUB_MAP 사전 대조 및 한글 변환
                    mapped_info = YOLO_TO_AIHUB_MAP.get(str(raw_label)) or YOLO_TO_AIHUB_MAP.get(str(cls_id))
                    
                    if mapped_info:
                        korean_name = mapped_info["korean_name"]
                        is_normal = mapped_info["is_normal"]
                    else:
                        # 맵에 없을 경우 원본 라벨 사용
                        korean_name = raw_label
                        is_normal = True if "normal" in raw_label.lower() else False
                        
                    detected_labels.append({
                        "name": korean_name,
                        "confidence": f"{conf:.2%}",
                        "is_normal": is_normal
                    })
                    
                    if not is_normal:
                        is_normal_status = False
                        
            # 탐지 결과 메타데이터 병합
            cv_data["ANNOTATION_INFO"] = annotations
            cv_data["IMAGE"]["ANNOTATION_COUNT"] = len(annotations)
            
            # 주된 병해충명 혹은 상태 결정 (가장 스코어가 높거나 첫번째 병해 기록 적용)
            disease_candidates = [l for l in detected_labels if not l["is_normal"]]
            if disease_candidates:
                cv_data["META"]["DBYHS_SPCHCKN"] = disease_candidates[0]["name"]
                cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = False
            else:
                normal_candidates = [l for l in detected_labels if l["is_normal"]]
                cv_data["META"]["DBYHS_SPCHCKN"] = normal_candidates[0]["name"] if normal_candidates else "정상"
                cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = True
                
        else:
            # YOLO 미설치/가중치 부재 시 데모 시뮬레이션
            st.info("💡 YOLO 모델이 부재하여 모의 데이터로 연동 테스트를 진행합니다.")
            st.image(image, caption="업로드된 이미지 (시뮬레이션 진단)", width="stretch")
            detected_labels = [
                {"name": "세균갈색무늬병", "confidence": "89.5%"}
            ]
            is_normal_status = False
            cv_data["META"]["DBYHS_SPCHCKN"] = "세균갈색무늬병"
            cv_data["META"]["DBYHS_NORMALITY_ALTERNATIVE"] = False
            cv_data["IMAGE"]["ANNOTATION_COUNT"] = 1
            cv_data["ANNOTATION_INFO"] = [{
                "ID": 1,
                "BOUNDING_BOX_X_COORDINATE": 200,
                "BOUNDING_BOX_Y_COORDINATE": 1200,
                "BOUNDING_BOX_WIDTH": 300,
                "BOUNDING_BOX_HEIGHT": 300
            }]
            
        # 화면 출력
        st.write("**실시간 탐지 라벨:**")
        for item in detected_labels:
            st.write(f"- 🔍 **{item['name']}** (신뢰도: {item['confidence']})")
            
        with st.expander("🔍 연동 중인 최종 JSON 메타데이터 확인"):
            st.json(cv_data)
    else:
        cv_data = None

with col2:
    st.subheader("📊 2. AI 환경 진단 및 컨설팅 피드백")
    
    if uploaded_file is not None and cv_data is not None:
        meta = cv_data.get("META", {})
        
        # 품종 및 데이터셋 요약
        st.info("**🍄 분석 대상 품종:** 느타리버섯 (Oyster Mushroom)")
        
        # 진단 결과 시각화
        is_normal = meta.get("DBYHS_NORMALITY_ALTERNATIVE", True)
        disease_name = meta.get("DBYHS_SPCHCKN", "정상")
        
        if is_normal:
            st.success(f"✅ **진단 결과:** 정상 생육 상태입니다. (현재 상태: {disease_name})")
        else:
            st.error(f"🚨 **진단 결과:** 병해 발생 우려 ({disease_name})")
            
        # 메트릭스 위젯
        m1, m2, m3 = st.columns(3)
        m1.metric("실내 온도 (Temperature)", f"{meta.get('TEMPERATURE')} °C")
        m2.metric("실내 습도 (Humidity)", f"{meta.get('HUMIDITY')} %")
        m3.metric("CO2 농도 (Carbon Dioxide)", f"{meta.get('CARBON_DIOXIDE')} ppm")
        
        # AI 처방 받기 버튼
        if st.button("🤖 AI 진단 처방 받기 (GPT-4o-mini)"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
            else:
                with st.spinner("스마트팜 환경 제어 솔루션 분석 중..."):
                    try:
                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0.4,
                            openai_api_key=openai_api_key
                        )
                        
                        prompt_template = ChatPromptTemplate.from_messages([
                            ("system", (
                                "당신은 버섯 스마트팜 재배 환경 및 병해충 방제 전문가입니다. "
                                "제공받은 센서 데이터(온도, 습도, 이산화탄소 등)와 진단 메타데이터를 정밀 대조하여 실질적이고 과학적인 해결책을 처방하세요."
                            )),
                            ("user", (
                                "버섯 스마트팜 실시간 측정 메타데이터는 다음과 같습니다:\n"
                                "```json\n{metadata_json}\n```\n\n"
                                "이 데이터를 분석하여 아래 질문에 전문적으로 답해주세요:\n\n"
                                "1. **센서 데이터 환경 진단**: 현재 온도({temp}°C), 습도({humidity}%), 이산화탄소({co2}ppm) 수치가 해당 품종({category_name})의 정상 생육 생태 환경(예: 양송이, 느타리 등의 표준 최적 환경 규격)에 적합한지 비교 분석해주세요.\n"
                                "2. **병해충({disease}) 원인 분석**: 현재 진단된 '{disease}'의 발생 원인을 현재의 온습도/이산화탄소 수치와 연관 지어 설명하고, 이것이 환경 제어 미흡에 따른 것인지 규명해주세요.\n"
                                "3. **스마트팜 제어 조치 처방**: 현재의 이병 상태를 해결하거나 추가 확산을 막기 위해 센서 수치(습도 낮추기, 환기 가동 등)를 어떻게 긴급 제어해야 하는지 명확한 가이드를 픽스해 조언해주세요."
                            ))
                        ])
                        
                        chain = prompt_template | llm | StrOutputParser()
                        
                        feedback = chain.invoke({
                            "metadata_json": json.dumps(cv_data, indent=2, ensure_ascii=False),
                            "temp": meta.get("TEMPERATURE"),
                            "humidity": meta.get("HUMIDITY"),
                            "co2": meta.get("CARBON_DIOXIDE"),
                            "category_name": "느타리버섯",
                            "disease": disease_name
                        })
                        
                        st.markdown("---")
                        st.markdown("### 📋 AI 종합 처방 리포트")
                        st.markdown(feedback)
                        
                    except Exception as e:
                        st.error(f"AI 가이드를 생성하는 중 에러가 발생했습니다: {e}")
    else:
        st.info("👈 왼쪽 영역에서 이미지를 먼저 업로드해 주세요. 이미지가 로드되면 실시간 진단 대시보드와 AI 컨설팅 결과가 나타납니다.")
