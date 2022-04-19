from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

x = [3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4,
     38.6, 50.2, 62.9, 76.0, 92.0, 105.7, 122.8, 131.7,
     150.7, 179.3, 203.2, 226.5, 248.7, 281.4]

t = np.arange(1790, 2010, 10)
t = (t - 1790) / 10

n = len(x)

# delta_x / delta_t
dxk = np.zeros(n)
dxk[0] = (-3 * x[0] + 4 * x[1] - x[2]) / 20
dxk[n-1] = (x[n-3] - 4 * x[n-2] + 3 * x[n-1]) / 20
for i in range(1, n-1):
    dxk[i] = (x[i+1] - x[i-1]) / 20

# y = delta_x / (delta_t * x)
y = dxk / x

data = np.array(x).reshape(-1, 1)
reg = linear_model.LinearRegression()
reg.fit(data, y)

# y = r - (r/xm)x
r = reg.intercept_
xm = -(r / reg.coef_)

print(r, xm)

y_pred = reg.predict(data)
y_pred *= 10


fig, ax = plt.subplots()
ax.plot(x, y*10, 'o', x, y_pred, '-')
plt.show()
