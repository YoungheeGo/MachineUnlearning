# MachineUnlearning
- 2023.11
- CIFAR10을 mock 객체로 하여 machine unlearning 진행
- 단, 기본모델은 Resnet-18을 사용하였고, unlearning기법으로 knowledge distillation을 사용함.
- teacher network(retain_data+forget_data)와 student network(retain_data)의 KL-divergence loss를 줄이는 방향으로 student network학습.


- private rank: 2, public rank:1  solution
  - https://www.kaggle.com/competitions/neurips-2023-machine-unlearning/discussion/458721
# 1등 솔루션
- self supervised contrastive learning 알고리즘을 사용하였다.
  - Positive 샘플들과 그들의 향상된 샘플들은 더 가깝게 하고, positive 샘플들과 모든 샘플들은 더 멀게 만든다.
  - 이 사람의 가정은, forget sample들이 feature space에서 가능한 uniform해야한다. -> forget 샘플과 retain 샘플들 사이의 거리를 더 멀리하기 위해서.
- Loss fuction 정의
  - forget sample X, positive 쌍의 enhanced version x'일때, retain batch에 있는 모든 샘플들은 negative pair
    
    $l_i = -log (e^{sim(x,x')/\tau}/sum_{batch2}(e^{sim(x,x')/\tau}))$

    $L_{forget}=1/batchsize (sum_{i=1}^{batch1}(l_i))$

- 새롭게 Self Supervised Contrastive Learning 알고리즘을 알게 되었다!

끝.
