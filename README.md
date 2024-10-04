# Highly Adaptive Ridge

Usage:

```
from kernel_ridge import HighlyAdaptiveRidgeCV, MixedSobolevRidgeCV 
from sklearn.model_selection import train_test_split

X, X_, Y, Y_ = train_test_split(X_data, Y_data, test_size=0.2)

har = HighlyAdaptiveRidgeCV() # or MixedSobolevRidgeCV()
har.fit(X,Y)
har.predict(X_)
```

See [paper](https://arxiv.org/pdf/2410.02680).
