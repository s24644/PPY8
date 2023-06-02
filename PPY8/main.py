import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
url = "./iris.data"

headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv(url, names=headers)
print(df.shape)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
leave_one_out = LeaveOneOut()
scores = cross_val_score(perceptron, X_train, y_train, cv=leave_one_out, scoring="accuracy")
print(scores)

accuracy = sum(scores) / len(scores)
print(accuracy)

# result_df = pd.DataFrame(X_test)
#
# fig, ax = plt.subplots(figsize=(100,5))
# ax.axis('tight')
# ax.axis('off')
# the_table = ax.table(cellText=result_df.values, colLabels=result_df.columns, loc='center')
#
# pp = PdfPages("wyniki.pdf")
# pp.savefig(fig, bbox_inches='tight')
# pp.close()

