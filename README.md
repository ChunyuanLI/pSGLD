# pSGLD
Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD)

## Simulation (2D Gaussian Example in Figure 1 of the paper)
- Simulation 1 provides _Average Absolute Error of Sample Covariance_ vs _AutoCorrelation Time (ACT)_
- Simulation 2 provides first 600 samples from SGLD and pSGLD

<img src="/simulation/2D/figure/pSGLD.png" data-canonical-src="/simulation/2D/figure/pSGLD.png" width="460" height="250" />

## Experiments on Deep Neural Networks (Keep updating)
- Start to run 'test_FNN_mnist.m' to test a 2-layer FNN with 400 hidden units each . 
- You may also modify line 'linSizes  = [400 400 data.outSize]' to other configurations. 


## Citation
Please cite our AAAI paper if it helps your research:

	@inproceedings{pSGLD_AAAI2016,
	  title={Preconditioned stochastic gradient Langevin dynamics for deep neural networks},
	  author={Li, Chunyuan and Chen, Changyou and Carlson, David and Carin, Lawrence},
	  booktitle={AAAI},
	  Year  = {2016}
	}
