import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="Healthcare Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """헬스케어 데이터 로드 및 전처리"""
    try:
        df = pd.read_csv('d:/장진/ai/windsurf/pandas/healthcare_dataset.csv')
        df_processed = df.copy()
        
        # 날짜 처리 및 입원 기간 계산
        df_processed['Date of Admission'] = pd.to_datetime(df_processed['Date of Admission'])
        df_processed['Discharge Date'] = pd.to_datetime(df_processed['Discharge Date'])
        df_processed['Length of Stay'] = (df_processed['Discharge Date'] - df_processed['Date of Admission']).dt.days
        
        # 타겟 변수 생성
        avg_stay = df_processed['Length of Stay'].mean()
        df_processed['Readmission_Risk'] = (df_processed['Length of Stay'] > avg_stay).astype(int)
        df_processed['Treatment_Outcome'] = df_processed['Test Results'].map({
            'Normal': 0, 'Abnormal': 1, 'Inconclusive': 2
        })
        
        # 범주형 변수 인코딩
        categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 
                              'Hospital', 'Insurance Provider', 'Admission Type', 'Medication']
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
        
        return df_processed, label_encoders
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None

@st.cache_resource
def train_models(df_processed):
    """머신러닝 모델 훈련"""
    # 특성 선택
    all_features = ['Age', 'Gender_encoded', 'Blood Type_encoded', 'Medical Condition_encoded', 
                   'Length of Stay', 'Doctor_encoded', 'Hospital_encoded', 'Insurance Provider_encoded', 
                   'Billing Amount', 'Admission Type_encoded', 'Medication_encoded']
    
    X = df_processed[all_features]
    y_billing = df_processed['Billing Amount']
    y_readmission = df_processed['Readmission_Risk']
    y_treatment = df_processed['Treatment_Outcome']
    
    # 데이터 분할
    X_train, X_test, y_billing_train, y_billing_test = train_test_split(X, y_billing, test_size=0.2, random_state=42)
    _, _, y_readmission_train, y_readmission_test = train_test_split(X, y_readmission, test_size=0.2, random_state=42)
    _, _, y_treatment_train, y_treatment_test = train_test_split(X, y_treatment, test_size=0.2, random_state=42)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 훈련
    rf_billing = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_billing.fit(X_train_scaled, y_billing_train)
    y_billing_pred = rf_billing.predict(X_test_scaled)
    billing_r2 = r2_score(y_billing_test, y_billing_pred)
    billing_rmse = np.sqrt(mean_squared_error(y_billing_test, y_billing_pred))
    
    rf_readmission = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_readmission.fit(X_train_scaled, y_readmission_train)
    y_readmission_pred = rf_readmission.predict(X_test_scaled)
    readmission_acc = accuracy_score(y_readmission_test, y_readmission_pred)
    
    rf_treatment = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_treatment.fit(X_train_scaled, y_treatment_train)
    y_treatment_pred = rf_treatment.predict(X_test_scaled)
    treatment_acc = accuracy_score(y_treatment_test, y_treatment_pred)
    
    return {
        'billing': {'model': rf_billing, 'r2': billing_r2, 'rmse': billing_rmse},
        'readmission': {'model': rf_readmission, 'accuracy': readmission_acc},
        'treatment': {'model': rf_treatment, 'accuracy': treatment_acc},
        'scaler': scaler,
        'features': all_features,
        'test_data': {
            'X_test': X_test_scaled, 'y_billing_test': y_billing_test, 'y_readmission_test': y_readmission_test,
            'y_treatment_test': y_treatment_test, 'y_billing_pred': y_billing_pred,
            'y_readmission_pred': y_readmission_pred, 'y_treatment_pred': y_treatment_pred
        }
    }

def main():
    # 사이드바
    st.sidebar.title("🏥 Healthcare Analytics")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.selectbox(
        "메뉴 선택",
        ["📊 데이터 개요", "📈 탐색적 데이터 분석", "🤖 예측 모델", "📋 모델 성능", "🔮 예측 시뮬레이션"]
    )
    
    # 데이터 로드
    df_processed, label_encoders = load_data()
    if df_processed is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    models = train_models(df_processed)
    
    st.title("🏥 Healthcare Data Analysis Dashboard")
    st.markdown("---")
    
    if menu == "📊 데이터 개요":
        show_data_overview(df_processed)
    elif menu == "📈 탐색적 데이터 분석":
        show_eda(df_processed)
    elif menu == "🤖 예측 모델":
        show_prediction_models(models)
    elif menu == "📋 모델 성능":
        show_model_performance(models)
    elif menu == "🔮 예측 시뮬레이션":
        show_prediction_simulation(df_processed, models, label_encoders)

def show_data_overview(df):
    """데이터 개요 페이지"""
    st.header("📊 데이터 개요")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 환자 수", f"{len(df):,}명")
    with col2:
        st.metric("평균 나이", f"{df['Age'].mean():.1f}세")
    with col3:
        st.metric("평균 청구 금액", f"${df['Billing Amount'].mean():,.0f}")
    with col4:
        st.metric("평균 입원 기간", f"{df['Length of Stay'].mean():.1f}일")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 데이터 정보")
        st.write(f"**데이터 크기:** {df.shape[0]:,}행 × {df.shape[1]}열")
        st.write(f"**결측치:** {df.isnull().sum().sum()}개")
        st.write(f"**중복 데이터:** {df.duplicated().sum()}개")
    
    with col2:
        st.subheader("🔍 데이터 미리보기")
        st.dataframe(df.head(10))

def show_eda(df):
    """탐색적 데이터 분석 페이지"""
    st.header("📈 탐색적 데이터 분석")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_age = px.histogram(df, x='Age', title='나이 분포', nbins=20)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        fig_billing = px.histogram(df, x='Billing Amount', title='청구 금액 분포', nbins=20)
        st.plotly_chart(fig_billing, use_container_width=True)
    
    with col3:
        fig_stay = px.histogram(df, x='Length of Stay', title='입원 기간 분포', nbins=15)
        st.plotly_chart(fig_stay, use_container_width=True)

def show_prediction_models(models):
    """예측 모델 페이지"""
    st.header("🤖 예측 모델")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 청구 금액 예측")
        st.metric("R² Score", f"{models['billing']['r2']:.3f}")
        st.metric("RMSE", f"${models['billing']['rmse']:,.0f}")
    
    with col2:
        st.subheader("⚠️ 재입원 위험 예측")
        st.metric("정확도", f"{models['readmission']['accuracy']:.3f}")
    
    with col3:
        st.subheader("🔬 치료 결과 예측")
        st.metric("정확도", f"{models['treatment']['accuracy']:.3f}")

def show_model_performance(models):
    """모델 성능 페이지"""
    st.header("📋 모델 성능 평가")
    
    test_data = models['test_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            x=test_data['y_billing_test'], 
            y=test_data['y_billing_pred'],
            title=f"실제 vs 예측 청구 금액 (R² = {models['billing']['r2']:.3f})"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        cm_readmission = confusion_matrix(test_data['y_readmission_test'], test_data['y_readmission_pred'])
        fig_cm = px.imshow(cm_readmission, text_auto=True, 
                          title=f"재입원 위험 혼동 행렬 (정확도 = {models['readmission']['accuracy']:.3f})")
        st.plotly_chart(fig_cm, use_container_width=True)

def show_prediction_simulation(df, models, label_encoders):
    """예측 시뮬레이션 페이지"""
    st.header("🔮 예측 시뮬레이션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 환자 정보 입력")
        age = st.slider("나이", min_value=18, max_value=100, value=45)
        gender = st.selectbox("성별", df['Gender'].unique())
        blood_type = st.selectbox("혈액형", df['Blood Type'].unique())
        medical_condition = st.selectbox("의학적 상태", df['Medical Condition'].unique())
        length_of_stay = st.slider("입원 기간 (일)", min_value=1, max_value=30, value=7)
    
    with col2:
        st.subheader("🎯 예측 결과")
        
        if st.button("예측 실행", type="primary"):
            try:
                # 입력 데이터 준비 (간소화)
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender_encoded': [label_encoders['Gender'].transform([gender])[0]],
                    'Blood Type_encoded': [label_encoders['Blood Type'].transform([blood_type])[0]],
                    'Medical Condition_encoded': [label_encoders['Medical Condition'].transform([medical_condition])[0]],
                    'Length of Stay': [length_of_stay],
                    'Doctor_encoded': [0], 'Hospital_encoded': [0], 'Insurance Provider_encoded': [0],
                    'Billing Amount': [df['Billing Amount'].mean()],
                    'Admission Type_encoded': [0], 'Medication_encoded': [0]
                })
                
                input_scaled = models['scaler'].transform(input_data)
                
                billing_pred = models['billing']['model'].predict(input_scaled)[0]
                readmission_pred = models['readmission']['model'].predict(input_scaled)[0]
                treatment_pred = models['treatment']['model'].predict(input_scaled)[0]
                
                st.metric("💰 예상 청구 금액", f"${billing_pred:,.0f}")
                
                risk_level = "높음" if readmission_pred == 1 else "낮음"
                st.metric("⚠️ 재입원 위험", risk_level)
                
                treatment_labels = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}
                st.metric("🔬 예상 치료 결과", treatment_labels[treatment_pred])
                
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
