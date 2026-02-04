{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7cb67d76-ea9e-4f9c-8e7e-d5fa9a2f30fe",
   "metadata": {},
   "source": [
    "# Linear Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cce27a31-8574-48a5-a957-76fd4bdfedb5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "64d26d4b-e50f-477c-b9bc-641469264c21",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_logistic_regression(df):\n",
    "    X = df.drop(\"Churn\", axis=1).values\n",
    "    y = df[\"Churn\"].values\n",
    "\n",
    "    X_train, X_test, y_train, y_test = train_test_split(\n",
    "        X, y, test_size=0.2, random_state=42, stratify=y\n",
    "    )\n",
    "\n",
    "    scaler = StandardScaler()\n",
    "    X_train = scaler.fit_transform(X_train)\n",
    "    X_test = scaler.transform(X_test)\n",
    "\n",
    "    model = LogisticRegression(\n",
    "        max_iter=1000,\n",
    "        class_weight=\"balanced\"\n",
    "    )\n",
    "\n",
    "    model.fit(X_train, y_train)\n",
    "\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    results = {\n",
    "        \"accuracy\": accuracy_score(y_test, y_pred),\n",
    "        \"confusion_matrix\": confusion_matrix(y_test, y_pred),\n",
    "        \"report\": classification_report(y_test, y_pred)\n",
    "    }\n",
    "\n",
    "    return results\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d94f5aeb-0332-4ff8-a6ee-deea946787f9",
   "metadata": {},
   "source": [
    "# End"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
