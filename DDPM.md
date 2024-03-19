# Denoising Diffusion Probabilistic Models  
## Introduction  
<<이 논문에서는 더 진전된 디퓨전 확률 모델을 제안>>   
- 디퓨전 모델은 유한 시간 후에 데이터와 일치하는 샘플을 생성하기 위해 번형 추론을 사용하여 학습된 __매개변수화된 마코프 체인__  
   *마코프 체인 : 특정 상태의 확률은 오직 과거의 상태에 의존하는 성질 가진 이산 확률과정*  
                  *즉 이전의 샘플링이 현재 샘플링에 영향을 미치는 $p(x|x)$ 형식을 의미*  
- 이 체인의 이행은 신호가 파괴될 때까지 샘플링 반대 방향으로 데이터에 노이즈를 점진적으로 추가하는 마르코프 체인인 diffusion 프로세스를 reverse 시키는 것으로 학습
- 이 diffusion model에서의 한 방향에 대해서는 주어진 이미지에 작은 가우시안 노이즈를 점진적으로 계속 더해서 완전히 이미지가 destroy되게 하는 과정을 의미
- Diffusion model 은 Generative model로서 학습된 데이터의 패턴을 생성해내는 역할을 함
- 패턴 생성 과정을 학습하기 위해 고의적으로 패턴을 무너트리고(Noising == Diffusion process), 이를 다시 복원하는 조건부 pdf를 학습함(Denoising == Reverse process)  
 ![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/879e7747-522d-42fa-b1ef-6b9fb9e38599)  
 
## Background  
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/e70db500-6f68-407c-9053-d6c9c407aa7d)
### Forward process(diffusion process) : 주어진 이미지 $x0$에서 서서히 gaussian noise를 추가하는 과정 $q$   
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/79b9b8a6-a49f-4467-896a-df9375813513)  
- $q(x_t∣x_t−1)$ : $x_t−1$에 noise를 적용해 $x_t$을 만드는 것
-  $x_T$ : 완전히 destroy 된 형태로 이는 normal distribution $N(x_T;0,I)$을 따른다.  
-  주입되는 gaussian noise 크기는 사전적으로 정의되고 이를 $βt$로 표기  
-  $βt$의 사전적 정의(scheduling)가 고려하는 3가지
    - Linear shecdule
    - Quad schedule
    - Sigmoid schedule

**diffusion process 는 conditional gaussian 의 joint-distribution으로서, $X0$를 조건부로 latent variables($X_1:r$)를 생성해내는 과정 $q$**
  - 가장 마지막 latent variable($X_T$ = $Z_T$)로 pure isotropic gaussian(방향에 따라 변하지 않는)을 획득
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/d5b5fcbb-4a1f-4f65-87c0-4b9f3fe5846c)  
  
### Reverse process(denoising process) : 주어진 이미지에서 gaussian noise를 점직적으로 제거해가며 특정한 패턴을 만들어가는 과정 $p$  
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/7d0189b8-425d-4e89-932c-29ea6845e5a4)  
  - **노란색 밑줄이 학습 대상(mean & variance function)**
  - diffusion process와 차이는 diffusion process에서 사전에 정의한 노이즈 크기인 베타에 의해서 모수인 평균과 분포가 정의가 되어 알고 있고 만들어 낼 수 있는 것이었지만 reverse process는 우리가 알지못하는 조건부 가우시안 분포라는 점  
  - 그래서 조건부 가우시안 분포의 모수인 평균과 분산을 학습해야함.  

- $p(x_t-1∣x_t)$ : $x_t$에 noise를 걷어내 $x_t-1$을 만드는 것, diffusion process의 역 과정(Denoising)을 학습  
- $x_t$들은 서로 resolution이 같다.  
- $p_θ($x_0$)$ : $x_0$에 대한 확률 분포  
- $pθ(x0:T)dx1:T$ : (x1부터 xT까지)에 대한 결합 확률 분포와 x1부터 xT까지의 모든 숨겨진 변수에 대한 적분 ==> $pθ(x0)$을 계산하기 위해 데이터 $x0$의 확률 분포를 계산하는 것  
#### Reverse process loss   
- reverse process가 학습의 대상, 그래서 학습의 방향을 보여줄 수 있는 loss function 필요
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/a899b6f9-96f4-4c80-913a-7142620fa644)
- diffusion의 loss는 VAE와 비교 이해 가능
- diffusion은 매우 많은 수의 latent variables 을 만들어가는 마코프 체인이 diffusion에 존재하게 되는데 이러한 latent variables을 control 할 수 있는 loss term이 denoising process
  ![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/9a5602ef-6566-4394-b294-02ddd8d4a434)
  ![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/80225ee0-76f1-4aa6-bb33-5e196fa89cbd)  
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/e63ae0d1-d508-470a-a841-7762615ed82b)


## Denoising Diffusion Probabilistic Model  
### DDPM Loss  
![image](https://github.com/MINJEONG-L/Denoising-Diffusion-Probabilistic-Models/assets/82145878/749c182b-dcd3-4d3d-9540-6090ea652618)  
- 노란색 : ground truth
- 파란색 : 학습 대상임을 나타내는 세타가 붙은 엡실론  
- t라는 각 시점의 노이즈인 앱실론을 모델이 예측하도록하는 loss  
1. 학습 목적식에서 regularization term 제외  
   - regularization term 은 T시점의 latent variable이 특정한 가우시안 분포를 따르도록 강제하는 역할인데, 1,000 steps를 거쳐 noise를 주입 --> 굳이 regularization term을 사용해서 $βt$를 학습하지 않더라도 T시점의 latent variable이 isotropic gaussian과 매우 유사하게 형성됨.  
   - 실제로 특정한 범위에서 $βt$를 linear scheduling으로 얻어낸 T시점에서의 latent variable이 매우 isotropic한 gaussian 분포를 따르게 됨.  
   - **따라서 regularization term을 제외하고 noise 크기인 $β$도 학습 대상이 아닌 fixed noise scheduling으로 진행해도 필요한 isotropic gaussian을 획득 가능**  


