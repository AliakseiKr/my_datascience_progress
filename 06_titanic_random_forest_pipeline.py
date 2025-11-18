import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv(r"C:\Users\lesha\Desktop\data science\train.csv")

def get_title(name): 
    if "Mr." in name:
        return "Mr"
    elif "Mrs."  in name:
        return "Mrs"
    elif "Miss"  in name:
        return "Miss"
    elif "Master" in name:
        return "Master"
    else:  
        return "Other"


df["Title"]=df["Name"].apply(get_title)

title_map = {
    "Mr": 1,
    "Mrs": 2,
    "Miss": 3,
    "Master": 4,
    "Other": 5
}

df["Title"] = df["Title"].map(title_map)

#Data preparation
avg_age=df["Age"].mean()
df["Age"]=df["Age"].fillna(avg_age)

df["Sex"]=df["Sex"].map({"male":1, "female":0})

from sklearn.model_selection import train_test_split

#feature engineering
df["FamilySize"]=df["SibSp"]+df["Parch"]+1
df["IsAlone"]=(df["FamilySize"]==1).astype(int)

X = df[[ "Pclass", "Sex", "Age", "Fare", "FamilySize", "IsAlone", "Title"]]


y=df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

machine_learn_model=RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    max_features=3,
    random_state=42,
    class_weight="balanced"
)
machine_learn_model.fit(X_train, y_train)

#prediction
y_pred=machine_learn_model.predict(X_test)




#estimating of model's accuracy by accuracy score
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#estimating model's accuracy by confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))



