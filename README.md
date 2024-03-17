# MachineUnlearning
- 2023.11
- CIFAR10을 mock 객체로 하여 machine unlearning 진행
- 단, 기본모델은 Resnet-18을 사용하였고, unlearning기법으로 knowledge distillation을 사용함.
- teacher network(retain_data+forget_data)와 student network(retain_data)의 KL-divergence loss를 줄이는 방향으로 student network학습.


