### LMOCSO: Large-scale multi-objective competitive swarm optimizer

##### Reference: Tian Y, Zheng X, Zhang X, et al. Efficient large-scale multiobjective optimization based on a competitive swarm optimizer[J]. IEEE Transactions on Cybernetics, 2019, 50(8): 3696-3708.

##### LMOCSO is a multi-objective evolutionary algorithm (MOEA) using competitive swarm optimizer, shift-based density estimation, and RVEA environmental selection.

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| nobj      | The dimension of objective space (default=3)                 |
| eta_m     | Perturbance factor distribution index (default = 20)         |
| alpha     | The parameter to control the change rate of APD (default = 2) |
| nvar      | The dimension of decision space                              |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| V         | Reference vectors                                            |
| zmin      | Ideal points                                                 |
| loser     | The half population with smaller fitness value               |
| winner    | The half population with bigger fitness value                |
| pf        | Pareto front                                                 |

#### Test problem: DTLZ1

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = 100 \left[|x_M| + \sum_{x_i \in x_M}(x_i - 0.5)^2 - \cos(20\pi(x_i - 0.5)) \right] \\
	& \min \\
	& f_1(x) = \frac{1}{2}x_1x_2 \cdots x_{M - 1}(1 + g(x_M)) \\
	& f_2(x) = \frac{1}{2}x_1x_2 \cdots (1 - x_{M - 1})(1 + g(x_M)) \\
	& \vdots \\
	& f_{M - 1}(x) = \frac{1}{2}x_1(1 - x_2)(1 + g(x_M)) \\
	& f_M(x) = \frac{1}{2}(1 - x_1)(1 + g(x_M)) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 2000, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/LMOCSO/Pareto front.png)



