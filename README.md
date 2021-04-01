# Autoencoder

lib : tensorflow, keras

epochs = 240

batchsize = 8

model_input = train dataset(노이즈를 추가한 이미지 셋)
model_output = cleaned dataset(노이즈가 없는 원본 이미지 셋)

intput = test image

output = result image

이미지 사이즈 1024x1024

model
![image](https://user-images.githubusercontent.com/40592785/113293154-5f35b280-9330-11eb-82a8-783a20a98d81.png)

옵티마이저 optimizers.Adam()
손실함수 'MSE'
