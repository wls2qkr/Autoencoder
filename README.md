# Autoencoder

#### dataset
dataset : [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
DIV2K dataset -> image size 조절 -> original folder

#### add noise
original + nosie -> train (addnoise.py)

#### input
original folder (cleaned images)
train folder (noised images)
test folder (test images)

#### output
test_result (result images)

##### use version
Python 3.7, Anaconda 가상환경 사용

tensorflow 2.3

##### model
![image](https://user-images.githubusercontent.com/40592785/113293154-5f35b280-9330-11eb-82a8-783a20a98d81.png)



##### 모델 컴파일
```model.compile(optimizer=optimizers.Adam(), loss='MSE')```

optimizer(정규화함수) : Adam 옵티마이저 사용, [Keras Doc](https://keras.io/api/optimizers/)

loss(손실함수) : 평균 제곱 오차(Mean Squared Error), [Keras Doc](https://keras.io/api/losses/regression_losses/#meansquarederror-class)

자주 사용되는 MSE를 사용했다 모델의 출력 값과 사용자가 원하는 출력 값 사이의 거리 차이를 오차로 사용한다. 신경망 학습에서는 최적의 매개변수를 탐색할 때 손시함수의 값을 가능한 작게하는 매개변수 값을 찾는다. 이 때 매개변수의 미분(기울기)을 계산하고, 그 값을 토대로 매개변수 값을 갱신하는 과정을 반복한다.
