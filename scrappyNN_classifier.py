from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyNN:

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train


    def predict(self, X_test):
        predictions = []
        for row in X_test:
            predictions.append(self.closest(row))
        return predictions

    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist<best_dist:
                best_dist = dist
                best_index = i
        return Y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split(x,y,test_size = 0.5)

my_classifier = ScrappyNN()
my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test,predictions)
