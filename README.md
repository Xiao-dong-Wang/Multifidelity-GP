# MFGP

## About
Multi-fidelity  Gaussian Process

This is a probabilistic framework based on Gaussian process regression and nonlinear autoregressive schemes that is capable of learning complex nonlinear and space-dependent crosscorrelations between models of variable fidelity, and it can effectively safeguard against low-fidelity models that provide wrong trends. 

The regression results comparisons between conventional Gaussian process method and multi-fidelity Gaussian process method are shown here.

![image](https://github.com/Xiao-dong-Wang/Multifidelity-GP/blob/master/figures/test1.png)

![image](https://github.com/Xiao-dong-Wang/Multifidelity-GP/blob/master/figures/test2.png)

Codes reimplemented here is based on the idea from the following paper:

- P. Perdikaris, M. Raissi, A. Damianou, N. Lawrence, and G. E. Karniadakis, “Nonlinear information fusion algorithms for data-efficient multifidelity modelling,” *Proc. R. Soc. A*, vol. 473, no. 2198, p. 20160751, 2017.

## Usage
See *run.sh*.

```
python run.py test2.toml
```



## Dependencies:

Autograd: https://github.com/HIPS/autograd

Scipy: https://github.com/scipy/scipy
