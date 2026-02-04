{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "03abced7-79d6-4c19-9099-bd134fff86c6",
   "metadata": {},
   "source": [
    "# Neural Netowrk Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b7f3dcc1-5e21-4ded-9aa7-7754e8b16252",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import library\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "from torch.utils.data import TensorDataset, DataLoader, random_split\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e8ec4d0c-bc01-4b07-bcec-a52c18ee58d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "class ChurnNN(nn.Module):\n",
    "    def __init__(self, input_dim):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(input_dim, 16),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(16, 1)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.net(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "06321acd-9d0e-4de5-ab2e-8795ed5fbd67",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_neural_network(df, epochs=100):\n",
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
    "    X_train = torch.tensor(X_train, dtype=torch.float32)\n",
    "    X_test = torch.tensor(X_test, dtype=torch.float32)\n",
    "    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)\n",
    "    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)\n",
    "\n",
    "    train_ds = TensorDataset(X_train, y_train)\n",
    "    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)\n",
    "\n",
    "    model = ChurnNN(X_train.shape[1])\n",
    "    criterion = nn.BCEWithLogitsLoss()\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n",
    "\n",
    "    for _ in range(epochs):\n",
    "        for xb, yb in train_loader:\n",
    "            optimizer.zero_grad()\n",
    "            loss = criterion(model(xb), yb)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        probs = torch.sigmoid(model(X_test))\n",
    "        y_pred = (probs >= 0.5).float()\n",
    "\n",
    "    y_pred = y_pred.numpy()\n",
    "    y_test = y_test.numpy()\n",
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
   "id": "acb139db-6f83-4dc9-be27-096a7f657a7d",
   "metadata": {},
   "source": [
    "# End"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbc88912-874a-4d1c-9c20-4e90beb4b496",
   "metadata": {},
   "outputs": [],
   "source": []
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
