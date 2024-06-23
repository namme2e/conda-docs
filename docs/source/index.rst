import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 예제 데이터 생성
data = {
    '광고_유형': ['텍스트', '이미지', '동영상', '텍스트', '이미지'],
    '광고_길이': [30, 20, 60, 25, 15],
    '사용자_연령': [25, 30, 35, 28, 32],
    '성별': ['남', '여', '여', '남', '여'],
    '클릭_여부': [1, 0, 1, 0, 1]  # 1: 클릭, 0: 클릭하지 않음
}

df = pd.DataFrame(data)

# 범주형 변수 더미화
df = pd.get_dummies(df, columns=['광고_유형', '성별'])

# 입력 변수(X)와 목표 변수(y) 분리
X = df.drop('클릭_여부', axis=1)
y = df['클릭_여부']

# 데이터를 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀모델 초기화 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'정확도: {accuracy:.2f}')

# 분류 보고서 출력
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력
print('혼동 행렬:')
print(confusion_matrix(y_test, y_pred))
