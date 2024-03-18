# Denoising Diffusion Probabilistic Models  
## Introduction  
<<이 논문에서는 더 진전된 디퓨전 확률 모델을 제안>>   
- 디퓨전 모델은 유한 시간 후에 데이터와 일치하는 샘플을 생성하기 위해 번형 추론을 사용하여 학습된 __매개변수화된 마코프 체인__  
   *마코프 체인 : 특정 상태의 확률은 오직 과거의 상태에 의존하는 성질 가진 이산 확률과정*  
                  *즉 이전의 샘플링이 현재 샘플링에 영향을 미치는 $p(x|x)$ 형식을 의미*  
- 이 체인의 이행은 신호가 파괴될 때까지 샘플링 반대 방향으로 데이터에 노이즈를 점진적으로 추가하는 마르코프 체인인 diffusion 프로세스를 reverse 시키는 것으로 학습
- 이 diffusion model에서의 한 방향에 대해서는 주어진 이미지에 작은 가우시안 노이즈를 점진적으로 계속 더해서 완전히 이미지가 destroy되게 하는 과정을 의미  
 ![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/879e7747-522d-42fa-b1ef-6b9fb9e38599)  
 
## Background  
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/e70db500-6f68-407c-9053-d6c9c407aa7d)
### Forward process(diffusion process) : 주어진 이미지 $x0$에서 서서히 noise를 추가하는 과정 $q$   
- $q(xt∣xt−1)$ : $xt−1$에 noise를 적용해 $xt$을 만드는 것
-  $xT$ : 완전히 destroy 된 형태로 이는 normal distribution $N(xT;0,I)$을 따른다.
### Reverse process(denoising process) : 주어진 이미지에서 noise를 점직적으로 걷어내는 과정 $p$  
- $p(xt-1∣xt)$ : $xt$에 noise를 걷어내 $xt-1$을 만드는 것
- $xt$들은 서로 resolution이 같다.  
- $pθ(x0)$ : $x0$에 대한 확률 분포  
- $pθ(x0:T)dx1:T$ : (x1부터 xT까지)에 대한 결합 확률 분포와 x1부터 xT까지의 모든 숨겨진 변수에 대한 적분 ==> $pθ(x0)$을 계산하기 위해 데이터 $x0$의 확률 분포를 계산하는 것

/// 잠재 변수 모델  
    - 데이터를 생성하는 과정을 설명하기 위해 관찰되지 않는 변수를 사용하는 확률 모델  
    - 관찰 가능한 변수(데이터)와 관찰되지 않는 변수(잠재변수) 간의 관계를 나타내는데 사용
## Diffusion models and denoising autoencoders  


