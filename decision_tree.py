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
####################################################################################################
# Treinando um modelo via k-fold-cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Definir as features e a variável alvo
X = data.drop(['Class'], axis=1)  # Remove a coluna 'Class' de X
y = data['Class']

# # Inicializar o classificador de Árvore de Decisão
# tree_clf = DecisionTreeClassifier(random_state=42)

# # Realizar o K-Fold Cross Validation com 5 divisões -> usando acurácia como métrica
# # cv=5 indica que será usado o K-Fold com k=5
# scores = cross_val_score(tree_clf, X, y, cv=5)

# # Imprimir as métricas de cada fold e a média
# print("Scores de cada fold:", scores)
# print("Média dos scores:", scores.mean())

# Realizar o K-Fold Cross Validation com 5 divisões -> usando PR AUC como métrica
scores = cross_val_score(tree_clf, X, y, cv=5, scoring='average_precision')

# Imprimir os scores de cada fold e a média dos scores
print("PR AUC de cada fold:", scores)
print("Média dos PR AUC:", scores.mean())
####################################################################################################
# para conseguir visualizar as arvores treinadas, vamos montar os folds manualmente 
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# numero de folds desejado
folds = 3

# Definir o número de folds e inicializar a validação cruzada estratificada
skf = StratifiedKFold(n_splits = folds)
train_accuracies = []
train_pr_aucs = []
test_accuracies = []
test_pr_aucs = []
depths = []

# Loop pelos folds
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treinar uma árvore de decisão para o fold atual
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)

    # Fazer previsões e calcular as métricas no conjunto de treino
    y_train_pred = tree_clf.predict(X_train)
    y_train_proba = tree_clf.predict_proba(X_train)[:, 1]  # Probabilidades da classe positiva
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_pr_auc = average_precision_score(y_train, y_train_proba)

    # Fazer previsões e calcular as métricas no conjunto de teste (validação)
    y_test_pred = tree_clf.predict(X_test)
    y_test_proba = tree_clf.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    depth = tree_clf.get_depth()

    # Armazenar as métricas e a profundidade
    train_accuracies.append(train_accuracy)
    train_pr_aucs.append(train_pr_auc)
    test_accuracies.append(test_accuracy)
    test_pr_aucs.append(test_pr_auc)
    depths.append(depth)

    # Visualizar a árvore de decisão para o fold atual
    # print(f"\nFold {fold + 1}")
    # print(f"Acurácia no treino: {train_accuracy:.4f}")
    # print(f"PR AUC no treino: {train_pr_auc:.4f}")
    # print(f"Acurácia no teste: {test_accuracy:.4f}")
    # print(f"PR AUC no teste: {test_pr_auc:.4f}")
    # print(f"Profundidade da árvore: {depth}")
    
    # plt.figure(figsize=(12, 8))
    # plot_tree(tree_clf, filled=True, feature_names=X.columns,
    #           class_names=[str(cls) for cls in tree_clf.classes_],
    #           rounded=True, fontsize=10)
    # plt.title(f"Árvore de Decisão - Fold {fold + 1}")
    # plt.show()

# Calcular e mostrar as médias das métricas no conjunto de treino e de teste
print("\nMétricas médias no conjunto de treino:")
print(f"Acurácia média no treino: {np.mean(train_accuracies):.4f}")
print(f"Média da PR AUC no treino: {np.mean(train_pr_aucs):.4f}")

print("\nMétricas médias no conjunto de teste:")
print(f"Acurácia média no teste: {np.mean(test_accuracies):.4f}")
print(f"Média da PR AUC no teste: {np.mean(test_pr_aucs):.4f}")

print(f"\nProfundidade média das árvores: {np.mean(depths):.2f}")
####################################################################################################

# Testaremos também um modelo de ensemble, semelhante a uma random forest
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from scipy.stats import mode
import numpy as np

# Definir o número de folds e inicializar a validação cruzada estratificada
skf = StratifiedKFold(n_splits=5)
train_accuracies = []
train_pr_aucs = []
test_accuracies = []
test_pr_aucs = []
validation_accuracies = []
validation_pr_aucs = []
depths = []
all_train_preds = []
all_test_preds = []
all_val_preds = []

# Loop pelos folds
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Treinar uma árvore de decisão para o fold atual
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)

    # Fazer previsões e calcular as métricas no conjunto de treino
    y_train_pred = tree_clf.predict(X_train)
    y_train_proba = tree_clf.predict_proba(X_train)[:, 1]  # Probabilidades da classe positiva
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_pr_auc = average_precision_score(y_train, y_train_proba)

    # Fazer previsões e calcular as métricas no conjunto de teste (validação)
    y_test_pred = tree_clf.predict(X_test)
    y_test_proba = tree_clf.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_pr_auc = average_precision_score(y_test, y_test_proba)

    # Fazer previsões para o conjunto de validação (se tiver um conjunto de validação separado)
    # Para este exemplo, vamos assumir que o conjunto de validação é o próprio conjunto de teste
    val_accuracy = test_accuracy  # Usando o teste como "validação"
    val_pr_auc = test_pr_auc

    # Armazenar as previsões de treino, teste e validação
    all_train_preds.append(y_train_pred)
    all_test_preds.append(y_test_pred)
    all_val_preds.append(y_test_pred)

    # Armazenar as métricas e a profundidade
    train_accuracies.append(train_accuracy)
    train_pr_aucs.append(train_pr_auc)
    test_accuracies.append(test_accuracy)
    test_pr_aucs.append(test_pr_auc)
    validation_accuracies.append(val_accuracy)
    validation_pr_aucs.append(val_pr_auc)
    depths.append(tree_clf.get_depth())

    # Visualizar a árvore de decisão para o fold atual
    print(f"\nFold {fold + 1}")
    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"PR AUC no treino: {train_pr_auc:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    print(f"PR AUC no teste: {test_pr_auc:.4f}")
    print(f"Acurácia na validação: {val_accuracy:.4f}")
    print(f"PR AUC na validação: {val_pr_auc:.4f}")
    print(f"Profundidade da árvore: {tree_clf.get_depth()}")

# Combinar as previsões de todas as árvores (usando moda para classificação)
final_train_preds = mode(all_train_preds, axis=0).mode[0]
final_test_preds = mode(all_test_preds, axis=0).mode[0]
final_val_preds = mode(all_val_preds, axis=0).mode[0]

# Calcular os scores finais nos conjuntos combinados
final_train_accuracy = accuracy_score(y_train, final_train_preds)
final_test_accuracy = accuracy_score(y_test, final_test_preds)
final_val_accuracy = accuracy_score(y_test, final_val_preds)

final_train_pr_auc = average_precision_score(y_train, final_train_preds)
final_test_pr_auc = average_precision_score(y_test, final_test_preds)
final_val_pr_auc = average_precision_score(y_test, final_val_preds)

# Exibir as métricas médias para todos os folds
print("\nMétricas médias:")
print(f"Acurácia média no treino: {np.mean(train_accuracies):.4f}")
print(f"Média da PR AUC no treino: {np.mean(train_pr_aucs):.4f}")

print(f"Acurácia média no teste: {np.mean(test_accuracies):.4f}")
print(f"Média da PR AUC no teste: {np.mean(test_pr_aucs):.4f}")

print(f"Acurácia média na validação: {np.mean(validation_accuracies):.4f}")
print(f"Média da PR AUC na validação: {np.mean(validation_pr_aucs):.4f}")

print(f"\nProfundidade média das árvores: {np.mean(depths):.2f}")

# Exibir os scores finais após combinar as previsões
print("\nScores finais combinados:")
print(f"Acurácia final no treino: {final_train_accuracy:.4f}")
print(f"Acurácia final no teste: {final_test_accuracy:.4f}")
print(f"Acurácia final na validação: {final_val_accuracy:.4f}")

print(f"PR AUC final no treino: {final_train_pr_auc:.4f}")
print(f"PR AUC final no teste: {final_test_pr_auc:.4f}")
print(f"PR AUC final na validação: {final_val_pr_auc:.4f}")
####################################################################################################