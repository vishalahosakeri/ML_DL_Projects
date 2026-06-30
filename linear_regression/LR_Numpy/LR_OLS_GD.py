#Scaling or decreasing learning rate is important to avoid divergence. For ex, x =100, lr = 0.01, it diverges. i.e it updates becomes larger
# 100X larger than before. It overshoots

import numpy as np

class linear_regression:

    def __init__(self):
        self.w = None
        self.b = None
        self.lossHistory = []

    def fit_ols(self,x,y):
        if x.ndim == 1:
            x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        Xwb = np.hstack([np.ones((x.shape[0],1)),x])
        #ols : w = (X^T @ X ) inverse @ X^T @y
        wgt_bias = np.linalg.pinv(Xwb.T @ Xwb) @ Xwb.T @ y
        # print("wegt_bias:",wgt_bias)
        self.w = wgt_bias[1:]
        self.b = wgt_bias[0]

    def predict(self,X):
        if X.ndim == 1:
            X = X.reshape(-1,1)
        if self.w == None and self.b == None:
            raise ValueError("Model is not fit yet, call fit_ols or fit_gd")
        return X @ self.w + self.b

    def fit_gd(self, X, y, lr=0.01,verbose=False, epoch =1500):
        print("gradient descent:")
        if X.ndim == 1:
            X = X.reshape(-1,1)
        y = y.reshape(-1,1)
        n, p = X.shape
        # print(X)

        self.w = np.zeros((p,1))
        self.b = 0.0
        self.lossHistory = []

        #for only 50 epoch
        for i in range (1, epoch+1):
            y_predict = self.predict(X)     
            error = y_predict - y

            #gradients 
            dw = (X.T @ error)/n
            db = error.mean()

            self.w -= lr * dw
            self.b -= lr * db

            loss = ((error ** 2).mean()) / 2
            self.lossHistory.append(float(loss))

            if verbose == True and i % 100 == 0:
                print(f"epoch :{epoch},loss:{loss:.6f}")
        return self

    def evaluate(self,X,y):
         if X.ndim ==1 :
             X = X.reshape(-1,1)
         y = y.reshape(-1,1)

         y_pred = self.predict(X)
         error = y - y_pred

         mse = (error ** 2).mean()
         rmse = np.sqrt(mse)
         
         ss_res = (error ** 2).sum()
         ss_tot = ((y - y.mean())**2).sum()
         r2 = 1 - (ss_res/ss_tot)

         print(f"mse:{mse},rmse:{rmse},r2:{r2}")

        
lrObj = linear_regression()
np.random.seed(42)
x = np.linspace(1,100,100,dtype=int)
y = 2 * x + 1 + np.random.rand(len(x))
# print(x,y)

split = int(0.8 * len(x))
# print(split)


x_train = x[:split]
# compute ONLY on train
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean)/std
y_train = y[:split]

x_test = x[split:]
x_test = (x_test - mean)/std
y_test = y[split:]

# lr.fit_ols(x,y)
# print(lr.w, lr.b)
# print(lr.predict(np.array([9,3]).reshape(-1,1)))

lrObj.fit_gd(x_train,y_train)

print("Weight and bias",lrObj.w, lrObj.b)
print(lrObj.predict(np.array([9,3]).reshape(-1,1)))
lrObj.evaluate(x_test,y_test)





