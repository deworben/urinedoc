## Version 1
import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

df = pandas.read_csv("urine_data.csv")

# print(df)



dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()


## Version 2

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

df = pd.read_csv("urine_data.csv")
clf = RandomForestClassifier(max_depth=len(df), random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))


