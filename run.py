import numpy as np
import pandas as pn
import matplotlib.pyplot as pplot


class LineerRegresyon:
    def __init__(self,sabitdeger=True):
        self.dizi=None
        self.deger=sabitdeger
        self.denk=False

    def ConvertToNumpy(self,x):
        if type(x)==type(pn.DataFrame()) or type(x)==type(pn.Series()):
            return pn.DataFrame(x)
        if type(x)==type(np.array([1,2])):
            return x
        return np.array(x)

    def Transpoze(self,x):
        if x.ndim==1:
            x = x.reshape(-1,1)
        return x

    def ConvertToArray(self,x):
        x=self.ConvertToNumpy(x)
        x=self.Transpoze(x)
        return  x

    def DegerEkle(self,X):
        satir=X.shape[0]
        sabit=np.ones(satir).reshape(-1,1)
        return np.hstack((X,sabit))

    def Denklem(self,X,Y):
        X=self.ConvertToArray(X)
        Y=self.ConvertToArray(Y)

        if self.deger:
            X = self.DegerEkle(X)
        num3 = np.linalg.inv(np.dot(X.T, X))
        num5=np.dot(X.T,Y)
        self.dizi=np.dot(num3,num5)
        self.denk=True

    def TahminEt(self,X):
        if not self.denk:
            ValueError("Denk true olmalı.")
        X=self.ConvertToArray(X)
        if self.deger:
            X=self.DegerEkle(X)
        return  np.dot(X,self.dizi)


    def ToplamDeger(self,X,Y):
        X=self.ConvertToArray(X)
        Y=self.ConvertToArray(Y)
        Tahmin=self.TahminEt(X)
        return np.mean((np.array(Tahmin)-(np.array(Y)))**2)


Dframe=pn.read_csv("weights_heights.csv")
Dframe.head()
tahmin1=pn.Series(Dframe["Height"])
tahmin2=pn.Series(Dframe["Weight"])

newframe=pn.concat(objs=(tahmin1,tahmin2),axis=1)
newframe=newframe.astype(int)

islemDframe=newframe[0:int((len(newframe)*70)/100)]
testDframe=newframe[int((len(newframe)*70)/100):len(newframe)]

X = islemDframe["Height"]
Y = islemDframe["Weight"]


Lineer = LineerRegresyon(sabitdeger=True)
Lineer.Denklem(X,Y)
Lineer.TahminEt([70])
Lineer.dizi


X = pn.Series(testDframe["Height"])
Y = Lineer.TahminEt(testDframe["Height"]).tolist()
testDframe.plot.scatter(x="Height", y="Weight")
pplot.plot(X,Y,color="red")

Lineer.ToplamDeger(islemDframe["Height"],islemDframe["Weight"])




