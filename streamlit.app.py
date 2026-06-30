import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv

# dotenv 로드 (.env 파일에서 API 키 읽기)
load_dotenv()

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
    layout="centered"
)

st.title("🍄 느타리버섯 생육 진단 & AI 재배 가이드")
st.write("느타리버섯의 이미지를 업로드하면 생육 상태(성장 단계)와 병해충 감염 여부를 분석하고, 최적의 재배 관리 피드백을 제공합니다.")

# 1. 모델 로드 (캐싱 활용)
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

# 2. 이미지 업로드 영역
uploaded_file = st.file_uploader("느타리버섯 재배 베드 또는 배지 이미지를 업로드해주세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("업로드 이미지")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("생육 및 병해충 진단 (YOLO)")
        
        detected_conditions = []
        
        if yolo_model is not None:
            with st.spinner("이미지 분석 및 객체 탐지 중..."):
                results = yolo_model(image)
                
                # 예측 결과 이미지 렌더링
                annotated_img_array = results[0].plot()
                annotated_image = Image.fromarray(annotated_img_array[..., ::-1]) # BGR to RGB
                st.image(annotated_image, use_column_width=True)
                
                # 탐지 결과 메타데이터 수집
                boxes = results[0].boxes
                names = yolo_model.names
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = names[cls_id]
                    detected_conditions.append({
                        "name": class_name,
                        "confidence": f"{conf:.2%}"
                    })
        else:
            st.info("실제 YOLO 모델(models/best.pt)이 준비되면 분석 결과가 표시됩니다. (현재는 시뮬레이션 진단 데이터 제공)")
            # 디버깅/시뮬레이션용 예시 메타데이터 (예: 생육 상태 및 일부 병해충 감지 상태)
            detected_conditions = [
                {"name": "생육기 (Growing Stage)", "confidence": "98.2%"},
                {"name": "세균성갈색무늬병 의심 (Bacterial Brown Blotch Suspected)", "confidence": "82.4%"}
            ]
            st.image(image, use_column_width=True)
            
        # 탐지 결과 텍스트 요약
        if detected_conditions:
            st.write("**탐지 결과 메타데이터:**")
            for item in detected_conditions:
                st.write(f"- 🔎 **{item['name']}** (신뢰도: {item['confidence']})")
        else:
            st.write("감지된 생육 지표 또는 특이 사항이 없습니다.")

    # 3. LangChain & LLM 피드백 영역
    if detected_conditions:
        st.markdown("---")
        st.subheader("🤖 AI 느타리버섯 재배 컨설팅 피드백")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
        else:
            with st.spinner("AI 맞춤 재배 가이드 생성 중..."):
                try:
                    # 1) LLM 초기화
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0.5,
                        openai_api_key=openai_api_key
                    )
                    
                    # 2) 프롬프트 템플릿 작성
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", (
                            "당신은 느타리버섯(Pleurotus ostreatus) 재배 및 버섯 병해충 전문가입니다. "
                            "농가로부터 받은 생육 단계 및 병해충 탐지 메타데이터를 정밀 진단하여 실무적인 피드백을 제공해야 합니다."
                        )),
                        ("user", (
                            "YOLO 컴퓨터 비전 모델을 통해 느타리버섯 재배 베드에서 진단된 메타데이터 정보입니다:\n"
                            "{metadata}\n\n"
                            "위 감지 결과를 바탕으로 아래 내용에 대해 전문적이고 구체적으로 조언해주세요:\n"
                            "1. **생육 단계별 환경 관리 가이드**: 감지된 생육 단계에 적합한 생육 관리 제어 요령(온도, 습도, CO2 농도/환기량 등)을 알려주세요.\n"
                            "2. **병해충 예방 및 긴급 방제 방안**: 병해충 증상이 감지된 경우 그 발생 원인과 긴급 대처 방법(환경 관리 조절, 이병 배지 제거, 소독 등)을 처방해주세요.\n"
                            "3. **품질 향상을 위한 추가 재배 조언**: 수확량과 품질을 극대화하기 위한 재배 팁을 제시해주세요."
                        ))
                    ])
                    
                    # 3) 체인 구성
                    chain = prompt_template | llm | StrOutputParser()
                    
                    # 메타데이터를 문자열로 가공
                    metadata_str = "\n".join([f"- 탐지 지표: {c['name']} (정확도: {c['confidence']})" for c in detected_conditions])
                    
                    # 4) 실행 및 피드백 출력
                    feedback = chain.invoke({"metadata": metadata_str})
                    st.markdown(feedback)
                    
                except Exception as e:
                    st.error(f"AI 피드백 생성 중 에러가 발생했습니다: {e}")
