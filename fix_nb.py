import json

f = open('loan-prediction-w-various-ml-models.ipynb', 'r', encoding='utf-8')
d = json.load(f)
f.close()

for i, c in enumerate(d['cells']):
    if 'NBclassifier1 = CategoricalNB()' in ''.join(c.get('source', [])):
        c['source'] = [
            'try:\n',
            '    NBclassifier1 = CategoricalNB()\n',
            '    NBclassifier1.fit(X_train, y_train)\n',
            '    y_pred = NBclassifier1.predict(X_test)\n',
            '    print(classification_report(y_test, y_pred))\n',
            '    print(confusion_matrix(y_test, y_pred))\n',
            '    from sklearn.metrics import accuracy_score\n',
            '    NBAcc1 = accuracy_score(y_pred,y_test)\n',
            '    print("Categorical Naive Bayes accuracy: {:.2f}%".format(NBAcc1*100))\n',
            'except Exception as e:\n',
            '    print(f"CategoricalNB failed: {e}")\n',
            '    NBAcc1 = 0\n'
        ]

f = open('loan-prediction-w-various-ml-models.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
