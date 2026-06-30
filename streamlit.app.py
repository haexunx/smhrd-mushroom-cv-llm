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
    page_title="버섯 인식 및 AI 가이드",
    page_icon="🍄",
    layout="centered"
)

st.title("🍄 버섯 인식 & AI 피드백 시스템")
st.write("버섯 이미지를 업로드하면 YOLO 모델이 분석하여 정보를 탐지하고, AI(GPT-4o-mini)가 맞춤 가이드를 제공합니다.")

# 1. 모델 로드 (캐싱 활용)
@st.cache_resource
def load_yolo_model(model_path="models/best.pt"):
    if not YOLO_AVAILABLE:
        return None
    try:
        # 가중치 파일 경로가 유효한지 체크 후 로드
        if os.path.exists(model_path):
            return YOLO(model_path)
    except Exception as e:
        st.warning(f"YOLO 모델 로드 실패: {e}")
    return None

yolo_model = load_yolo_model()

# 2. 이미지 업로드 영역
uploaded_file = st.file_uploader("버섯 이미지를 업로드해주세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("업로드된 이미지")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("객체 탐지 (YOLO)")
        
        # YOLO 분석 수행 및 메타데이터 추출
        detected_mushrooms = []
        
        if yolo_model is not None:
            with st.spinner("이미지 분석 중..."):
                # YOLO 예측 실행
                results = yolo_model(image)
                
                # 예측된 결과 렌더링 (바운딩 박스 표시된 이미지)
                annotated_img_array = results[0].plot()
                annotated_image = Image.fromarray(annotated_img_array[..., ::-1]) # BGR to RGB
                st.image(annotated_image, use_column_width=True)
                
                # 메타데이터 수집 (탐지된 버섯 정보)
                boxes = results[0].boxes
                names = yolo_model.names
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = names[cls_id]
                    detected_mushrooms.append({
                        "name": class_name,
                        "confidence": f"{conf:.2%}"
                    })
        else:
            st.info("실제 YOLO 모델(models/best.pt)이 준비되면 감지 결과가 표시됩니다. (현재는 시뮬레이션 데이터 제공)")
            # 디버깅/시뮬레이션용 예시 메타데이터
            detected_mushrooms = [
                {"name": "광대버섯 (Amanita muscaria)", "confidence": "94.5%"}
            ]
            st.image(image, use_column_width=True)
            
        # 탐지 결과 텍스트 요약
        if detected_mushrooms:
            st.write("**탐지 결과 메타데이터:**")
            for item in detected_mushrooms:
                st.write(f"- 🍄 **{item['name']}** (신뢰도: {item['confidence']})")
        else:
            st.write("감지된 버섯이 없습니다.")

    # 3. LangChain & LLM 피드백 영역
    if detected_mushrooms:
        st.markdown("---")
        st.subheader("🤖 AI 버섯 가이드 및 대처 피드백")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            st.error("⚠️ `.env` 파일에 `OPENAI_API_KEY`를 설정해주세요.")
        else:
            with st.spinner("AI 피드백 생성 중..."):
                try:
                    # 1) LLM 초기화
                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0.7,
                        openai_api_key=openai_api_key
                    )
                    
                    # 2) 프롬프트 템플릿 작성
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", (
                            "당신은 버섯 분류 및 독버섯 식별 전문가입니다. "
                            "탐지된 버섯 정보를 기반으로 정확하고 유용한 피드백을 제공하세요. "
                            "식용 여부를 명시하고, 독버섯일 경우 위험성과 대처법을 경고하세요."
                        )),
                        ("user", (
                            "YOLO 모델로부터 추출된 탐지 메타데이터는 다음과 같습니다:\n"
                            "{metadata}\n\n"
                            "이 버섯에 대해 다음 정보를 일목요연하게 알려주세요:\n"
                            "1. 버섯의 주요 특징 및 생태 정보\n"
                            "2. 식용 가능 여부 및 주의사항\n"
                            "3. 독버섯일 경우 나타나는 중독 증상과 대처 방법"
                        ))
                    ])
                    
                    # 3) 체인 구성
                    chain = prompt_template | llm | StrOutputParser()
                    
                    # 메타데이터를 문자열 포맷팅
                    metadata_str = "\n".join([f"- 종류: {m['name']} (신뢰도: {m['confidence']})" for m in detected_mushrooms])
                    
                    # 4) 실행 및 피드백 출력
                    feedback = chain.invoke({"metadata": metadata_str})
                    st.markdown(feedback)
                    
                except Exception as e:
                    st.error(f"AI 가이드 생성 중 에러가 발생했습니다: {e}")
