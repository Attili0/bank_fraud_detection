# Configurando o token do kaggle
import os
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    # Caminho do diretório .kaggle
    kaggle_dir = Path("/root/.kaggle/")

    # Verifica se o diretório .kaggle já existe
    if not kaggle_dir.exists():
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        print("Diretório .kaggle criado.")
    else:
        print("Diretório .kaggle já existe.")

    src = Path("/content/kaggle.json")
    dest = Path("/root/.kaggle/kaggle.json")
    # Move o arquivo kaggle.json para o diretório .kaggle se ele existir
    if src.exists():
        os.rename(src, dest)
        print("Arquivo kaggle.json movido para ~/.kaggle")
    elif dest.exists():
        print("Arquivo kaggle.json já está no destino.")
    else:
        print("Arquivo kaggle.json não encontrado em /content, procurando no drive")
        # verifica se o drive está montado
        drive_path = Path("/content/drive")
        if not drive_path.exists():
            print("Google Drive não está montado.")
            return
        else:
            print("Google Drive está montado.")
            kaggle_token_drive = drive_path / "MyDrive/kaggle.json"
            if not kaggle_token_drive.exists():
                print("Arquivo kaggle.json não encontrado no Google Drive.")
                return
            else:
                print("Arquivo kaggle.json encontrado no Google Drive.")
                src = Path("/content/drive/MyDrive/kaggle.json")
                dest = Path("/root/.kaggle/kaggle.json")
                shutil.copy(str(src), str(dest))
                print("Arquivo kaggle.json copiado para /.kaggle")
                os.chmod(dest, 0o600)
                print("Permissões do arquivo kaggle.json definidas.")


# Chama a função para configurar as credenciais do Kaggle
setup_kaggle_credentials()
####################################################################################################
# baixando e instalando a biblioteca do kaggle e o dataset do creditcardfraud
import kagglehub
import pandas as pd

# !pip install -q kaggle
# !kaggle datasets download -d mlg-ulb/creditcardfraud
# !unzip creditcardfraud.zip -d /content/creditcardfraud

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

downloaded_files = os.listdir(path)

data = pd.read_csv(os.path.join(path, downloaded_files[0]))
data.head() # exibe as primeiras () linhas do arquivo
####################################################################################################
from sklearn.model_selection import train_test_split
# determinar conjuntos de treino e teste
# vamos usar todas as features exceto 'class' como X e 'class' como y
X = data.drop(['Class'], axis=1)  # Remove a 'Class' de X
y = data['Class']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################################################### MODELO DE ÁRVORE DE DECISÃO #########################################################
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Inicializar o classificador de Árvore de Decisão
tree_clf = DecisionTreeClassifier(random_state=42)

# Treinar o modelo com os dados de treinamento
tree_clf.fit(X_train, y_train)
####################################################################################################
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualizar a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(tree_clf, filled=True, feature_names=X.columns, 
        class_names=[str(cls) for cls in tree_clf.classes_], 
        rounded=True, fontsize=10)
plt.show()

depth = tree_clf.get_depth()
print(f"Pprofundidade da árvore de decisão: {depth}")
####################################################################################################
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Obtenha as probabilidades preditas para a classe positiva no conjunto de treino
y_train_probs = tree_clf.predict_proba(X_train)[:, 1]  # Probabilidade para a classe positiva

# Calcule precisão e recall para diferentes thresholds
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_probs)

# Calcule a AUC da curva Precision-Recall para o treino
train_pr_auc = auc(recall_train, precision_train)
print(f"Área sob a curva Precision-Recall no conjunto de treino: {train_pr_auc:.5f}")

# Obtenha as probabilidades preditas para a classe positiva no conjunto de teste
y_test_probs = tree_clf.predict_proba(X_test)[:, 1]  # Probabilidade para a classe positiva

# Calcule precisão e recall para diferentes thresholds
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_probs)

# Calcule a AUC da curva Precision-Recall para o teste
test_pr_auc = auc(recall_test, precision_test)
print(f"Área sob a curva Precision-Recall no conjunto de teste: {test_pr_auc:.5f}")
print()

# # Gerar e plotar a curva ROC para o conjunto de treino
# fpr, tpr, thresholds = roc_curve(y_train, y_train_probs)

# # Plotando a curva ROC para o conjunto de treino
# plt.figure(figsize=(5, 3))
# plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc_score(y_train, y_train_probs):.2f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Linha aleatória (AUC = 0.5)')
# plt.xlabel('Taxa de Falsos Positivos (FPR)')
# plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
# plt.title('Curva ROC - Conjunto de Treino')
# plt.legend(loc='lower right')
# print("Curva Precision-Recall - Conjunto de Treino")
# plt.show()

# # Gerar e plotar a curva ROC para o conjunto de teste
# fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)

# # Plotando a curva ROC para o conjunto de teste
# plt.figure(figsize=(5, 3))
# plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_test_probs):.2f})')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Linha aleatória (AUC = 0.5)')
# plt.xlabel('Taxa de Falsos Positivos (FPR)')
# plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
# plt.title('Curva ROC - Conjunto de Teste')
# plt.legend(loc='lower right')
# print("Curva Precision-Recall - Conjunto de Teste")
# plt.show()

# Plotar a curva Precision-Recall para o conjunto de treino
plt.figure(figsize=(5, 3))
plt.plot(recall_train, precision_train, color='blue', label=f'Curva Precision-Recall (AUC = {train_pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva Precision-Recall - Conjunto de Treino')
plt.legend(loc='lower left')
plt.show()

# Plotar a curva Precision-Recall para o conjunto de teste
plt.figure(figsize=(5, 3))
plt.plot(recall_test, precision_test, color='green', label=f'Curva Precision-Recall (AUC = {test_pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva Precision-Recall - Conjunto de Teste')
plt.legend(loc='lower left')
plt.show()

