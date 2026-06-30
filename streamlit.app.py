import streamlit as st
import os
import json
from PIL import Image
from dotenv import load_dotenv

# dotenv 로드 (.env 파일에서 API 키 읽기)
load_dotenv()

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 페이지 레이아웃 및 스타일 설정
st.set_page_config(
    page_title="느타리버섯 생육 진단 & AI 컨설턴트",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 스마트팜 버섯 병해 및 생육 상태 AI 진단 서비스")
st.write("버섯 이미지 분석에서 추출된 AI Hub 표준 메타데이터와 실시간 환경 제어 변수들을 바탕으로 AI(gpt-4o-mini)가 정밀 피드백을 제공합니다.")

# 1. 표준 메타데이터 기본 템플릿
BASE_CV_OUTPUT = {
  "INFO": {
    "DATASET_NAME": "양송이 병해",
    "DATASET_DETAIL": "(스마트팜 통합데이터_버섯)",
    "VERSION": "1.0",
    "LICENSE": "",
    "CREATE_DATE_TIME": "2021-12-06 16:25:55",
    "CONTRIBUTOR": "",
    "URL": "https://www.labelon.kr",
    "CATEGORY_NAME": "양송이"
  },
  "IMAGE": {
    "IMAGE_URL": "https://images.labelon.kr/2021/11/30/5b853f816c80457ea6dc8172ab059f88.jpg",
    "IMAGE_FILE_NAME": "양송이_생육실1_8_16344202.jpg",
    "WIDTH": 1080,
    "HEIGHT": 1920,
    "ANNOTATION_COUNT": 1
  },
  "ANNOTATION_INFO": [
    {
      "ID": 65225628,
      "BOUNDING_BOX_X_COORDINATE": 204,
      "BOUNDING_BOX_Y_COORDINATE": 1261,
      "BOUNDING_BOX_WIDTH": 295,
      "BOUNDING_BOX_HEIGHT": 342,
      "SEGMENTATION": None,
      "SEGMENTATION_AREA_TOTAL": None,
      "CROWDSOURSING_OPERATION_ALTERNATIVE": True
    }
  ],
  "META": {
    "DBYHS_SPCHCKN": "푸른곰팡이병",
    "DBYHS_NORMALITY_ALTERNATIVE": False,
    "IP_CAMERA_ID": 8,
    "WIND_SPEED": 0.0,
    "AIR_VELOCITY": 0.0,
    "TEMPERATURE": 18.2,
    "HUMIDITY": 97.6,
    "ILLUMINATION_INTENSITY": 0.0,
    "CARBON_DIOXIDE": 1156.0,
    "GUIDELINE": None,
    "IMAGE_CREATE_DATE": "2021-11-28",
    "IMAGE_CREATE_TIME": "15:08:11",
    "IMAGE_CREATE_DAY_OF_WEEK": "Sunday",
    "STIPE_LENGTH": None,
    "STIPE_THICKNESS": None,
    "PILEUS_DIAMETER": None,
    "PILEUS_THICKNESS": None,
    "GROSS_WEIGHT": None
  }
}

# UI 구성: 2단 컬럼 레이아웃
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📸 1. 이미지 업로드 및 센서 값 설정")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("버섯 이미지를 업로드해주세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    # 이미지가 업로드된 경우에만 데이터 세팅 슬라이더 노출
    if uploaded_file is not None:
        st.image(uploaded_file, caption="업로드된 버섯 이미지", use_column_width=True)
        
        st.markdown("---")
        st.markdown("### 🎛️ 실시간 센서 환경 변수 조절")
        st.write("진단 시뮬레이션을 위해 아래 센서 데이터를 조절해보세요.")
        
        # 온도, 습도, 이산화탄소 농도만 슬라이더로 수정 가능하도록 변경
        edited_temp = st.slider("실내 온도 (°C)", min_value=10.0, max_value=35.0, value=18.2, step=0.1)
        edited_humid = st.slider("실내 습도 (%)", min_value=50.0, max_value=100.0, value=97.6, step=0.1)
        edited_co2 = st.slider("이산화탄소 농도 (CO2, ppm)", min_value=300, max_value=3000, value=1156, step=10)
        
        # 기본 메타데이터 구조에 수정된 값 대입
        cv_data = BASE_CV_OUTPUT.copy()
        cv_data["META"]["TEMPERATURE"] = edited_temp
        cv_data["META"]["HUMIDITY"] = edited_humid
        cv_data["META"]["CARBON_DIOXIDE"] = edited_co2
        
        # 백그라운드용 완성된 JSON 문자열 정보
        with st.expander("🔍 완성된 전체 메타데이터(JSON) 확인"):
            st.json(cv_data)
    else:
        cv_data = None

with col2:
    st.subheader("📊 2. AI 환경 진단 및 컨설팅 피드백")
    
    # 이미지가 업로드된 상태에서만 진단 화면을 표시
    if uploaded_file is not None and cv_data is not None:
        info = cv_data.get("INFO", {})
        meta = cv_data.get("META", {})
        
        # UI에 추출된 품종 정보 요약 표시
        st.info(f"**🍄 분석 대상 품종:** {info.get('CATEGORY_NAME')} ({info.get('DATASET_NAME')})")
        
        # 정상 여부 및 병해충 분류 시각화
        is_normal = meta.get("DBYHS_NORMALITY_ALTERNATIVE", True)
        disease_name = meta.get("DBYHS_SPCHCKN", "정상")
        
        if is_normal:
            st.success("✅ **진단 결과:** 정상 생육 상태입니다.")
        else:
            st.error(f"🚨 **진단 결과:** 병해 발생 우려 ({disease_name})")
            
        # 설정된 센서 데이터 메트릭스 표시
        m1, m2, m3 = st.columns(3)
        m1.metric("조정된 온도 (Temperature)", f"{meta.get('TEMPERATURE')} °C")
        m2.metric("조정된 습도 (Humidity)", f"{meta.get('HUMIDITY')} %")
        m3.metric("조정된 CO2 농도", f"{meta.get('CARBON_DIOXIDE')} ppm")
        
        # LLM 피드백 호출 버튼
        if st.button("🤖 AI 진단 처방 받기 (GPT-4o-mini)"):
            openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
            else:
                with st.spinner("스마트팜 환경 제어 솔루션 분석 중..."):
                    try:
                        # 1) LLM 세팅
                        llm = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0.4,
                            openai_api_key=openai_api_key
                        )
                        
                        # 2) 프롬프트 템플릿 구성
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
                        
                        # 3) 체인 바인딩
                        chain = prompt_template | llm | StrOutputParser()
                        
                        # 4) 호출 파라미터 전달
                        feedback = chain.invoke({
                            "metadata_json": json.dumps(cv_data, indent=2, ensure_ascii=False),
                            "temp": meta.get("TEMPERATURE"),
                            "humidity": meta.get("HUMIDITY"),
                            "co2": meta.get("CARBON_DIOXIDE"),
                            "category_name": info.get("CATEGORY_NAME"),
                            "disease": disease_name
                        })
                        
                        st.markdown("---")
                        st.markdown("### 📋 AI 종합 처방 리포트")
                        st.markdown(feedback)
                        
                    except Exception as e:
                        st.error(f"AI 가이드를 생성하는 중 에러가 발생했습니다: {e}")
    else:
        # 이미지 업로드 유도 안내 메시지 표시
        st.info("👈 왼쪽 영역에서 이미지를 먼저 업로드해 주세요. 이미지가 로드되면 실시간 진단 대시보드와 AI 컨설팅 결과가 나타납니다.")
