import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

n_estimators = 100

train_df = pd.read_csv("C:/Users/Никита/elyra/data/features/pr_train.csv")
test_df = pd.read_csv("C:/Users/Никита/elyra/data/features/pr_test.csv")

target = 'Transported'
features = list(train_df.columns)
features.remove(target)

X_train = train_df[features]
y_train = train_df[[target]]
X_test = test_df[features]
y_test = test_df[[target]]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train.values.ravel())

y_train_pred = clf.predict(X_test)

with open('f1_score.txt', 'w') as the_file:
    the_file.write(str(f1_score(y_test, y_train_pred)))