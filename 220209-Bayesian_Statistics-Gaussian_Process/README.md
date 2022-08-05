# GP talk

Based on introductory machine learning lectures:

[Lecture 4: Estimating Probabilities from data](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote04.html)

[Lecture 8: Linear Regression](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote08.html)

[Lecture 15: Gaussian Process](http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html)

[Stanford lecture notes on GPs](http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf)

# Content

1. Goal: Estimating probabilities from data
1. Linear Regression
2. Bayesian linear regression
3. Gaussian processes
4. Kernel trick
5. Gaussian Process Kernels
6. Hyperoptimisation
7. Prediction
8. Example: CO2 concentrations



# Estimating probabilities from data

If we are provided with $P(X,Y)$ we can predict the most likely label for $\bf x$, formally $\argmax_yP(y|\mathbf{x})$. It is therefore worth considering if we can estimate $P(X,Y)$ directly from the training data. If this is possible (to a good approximation) we could then use the Bayes Optimial classifier in practice on our estimate of $P(X,Y)$. 

In fact, many supervised learning can be viewed as estimating $P(X,Y)$. Generally, they fall ito two categories:
- When we estimate $P(X,Y)=P(X|Y)P(Y)$, then we call it *generative learning*.
- When we estimate $P(X,Y)=P(Y|X)P(Y)$ which means determing $P(Y|X)$ directly, then we call it *discriminative learning*.

Most approaches nowadays are dicriminative.

Once we have a way to estimate the distribution of data, prediction becomes trivial. 

        So how can we estimate probability distributions from samples?


# Linear Regression

Assume data described by $f(x)=wx^T$, and try to find $w$. Of course the data is not on a perfect line, but Gaussianly distributed:
$$ 
f(x)=wx^T+\epsilon ,
$$
where $\epsilon$ is the variance.

There are may ways to determine $w$, one of those is maximum likelihood estimation (MLE):
$$
P(D|w)=\prod_{i=1}^{n}P(y_i|x_i;w), \quad D=\{(x_1,y_1),(x_2,y_2),....,(x_n,y_n)\},
$$
which essentialy asks the question "Which $w$ best describes our data?". 

Alternatively we can use MAP:
$$
P(w|D)\propto \frac{P(D|w)P(w)}{N},
$$
where $P(w)$ is a Gaussian prior and $N$ a normalization factor. Here we essentailly ask the question "given a set of data, what is the most likely set of parameters?". 

Considering MLE, a single data point can be described as
$$
\overbrace{P(y_1|x_1;w)}^{\rm Gaussian}=\mathcal{N}(w^Tx,\sigma^2I),
$$
which means that the distribution over each data point is described by a Gaussian distribution since the product of Gaussian distributions is a Gaussian distribution.

Given our data $D$, we determine a $w$, from which we can then provide a prediction $y$.

Observation: $w$ is only used to make a prediction. Once the prediction is made, it is irrelevant what $w$ was used in making the prediction (assuming $w$ was correct). So why do we model the prediction of a test point from the start, instead of first modelling the probability of $w$ and then making predictions with that. 

# Gaussian Process Regression

## Posterior Predictive Distribution

We want to predict $P(y|x)$ without making an assumption about the model. We simply ask "given a test point $x$, what is the distirbution of $y$?". In order to make predictions, we do need a model, but we can marganilize it out:
$$ 
\overbrace{P(x|y,D)}^{\rm Gaussian}
=\int_w P(y|x,D,w)dw=\int_w \overbrace{P(y|x,w)}^{\rm Gaussian} \overbrace{P(w|D)}^{\rm Gaussian}dw. 
$$
Here $P(y|x,w)$ is Gaussian, while 
$$ 
{P(w|D)}=\frac{\overbrace{P(D|w)}^{\rm Gaussian}\overbrace{P(w)}^{\rm Gaussian}}{N}, 
$$
is Gaussian. Where the conjugate prior $P(w)$ is chosen/defined such that it is Gaussian.

So what we have done is essentally is to allow us to make predictions without commiting to a single choice of $w$, but instead we consider all possible values of $w$. But $w$ resulting in a poor fit will hardly affect the prediction while $w$ that result in a good fit to the data have a large impact on the prediction. **SHOW POSSIBLE LINES TO DATA, DARKER COLOR FOR CLOSER FIT**

Unfortunately, the above is often intractable in closed form. However, for the special case of having a Gaussian likelihood and prior (those are the ridge regression assumptions), this expression is Gaussian and we can derive its mean and covariance. So,
$$
P(y_*|D,{\bf x})\sim \mathcal{N}(\mu_{y_*|D},\Sigma_{y_*|D}),
$$
where 
$$
\mu_{y_{*} \mid D}=K_{*}^{T}\left(K+\sigma^{2} I\right)^{-1} y
$$
and
$$
\Sigma_{y_{*} \mid D}=K_{* *}-K_{*}^{T}\left(K+\sigma^{2} I\right)^{-1} K_{*}.
$$
So, instead of doing MAP (as in ridge regression) let's model the entire distribution and let's forget about $w$ and the kernel trick by modelling $f$ directly (instead of $y$)!


## Gaussian Processes - Definition

To model this distribution we need a Kernel function
