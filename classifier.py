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
# Análises úteis do banco de dados
import numpy as np
import matplotlib.pyplot as plt
# o comando describe dá as estatísticas gerais por coluna
# data.drop(['Class'], axis=1).describe()

# colocando o ponto de referencia da feature time no dia e nao no evento 0.0
# essa conversão nao mostrou melhora nos resultados do modelo linear, mas ajuda na visualização grafica
# data['Time'] = (data['Time']/3600) % 24

# histogramas por coluna
# insira a feature estudada
feat = 'Time'
# determina os intervalos de x e y
amount_min = data[feat].min()
amount_max = data[feat].max()
amount_range = (amount_min, amount_max)

data_class_0_counts, _ = np.histogram(data[data['Class'] == 0][feat], bins=5, range=amount_range)
data_class_1_counts, _ = np.histogram(data[data['Class'] == 1][feat], bins=5, range=amount_range)
y_max = max(data_class_0_counts.max(), data_class_1_counts.max())

# Dados para os casos onde Class = 0
data_class_0 = data[data['Class'] == 0][feat]
# data_class_0 = data[data['Class'] == 0].drop([feat], axis=1)
data_class_0.hist(bins=5, range=amount_range, figsize=(14, 3), color='skyblue')
plt.suptitle("Transações legítimas")
# plt.ylim(0, y_max)
plt.show()

# Dados para os casos onde Class = 1
data_class_1 = data[data['Class'] == 1][feat]
# data_class_1 = data[data['Class'] == 1].drop([feat], axis=1)
data_class_1.hist(bins=5, range=amount_range, figsize=(14, 3), color='salmon')
plt.suptitle("Fraudes")
# plt.ylim(0, y_max)
plt.show()
####################################################################################################
# determinar conjuntos de treino e teste
from sklearn.model_selection import train_test_split
# vamos usar todas as features exceto 'class' como X e 'class' como y
X = data.drop(['Class'], axis=1)  # Remove a 'Class' de X
y = data['Class']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
####################################################################################################
########################################################### MODELO DE CLASSIFICADOR #########################################################
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
# Treinando o modelo
deg = 1        # grau do polinomio
pol = "cubic"   # tipo normal(normal) e cubic(nao usa termos retangulares)
reg = None    # regularização None, "l1", "l2"
solver_alg = "lbfgs"   # solver, para padrao -> "lbfg", para lasso -> "saga"
iter_limit = 10000  # numero maximo de iterações
clf_nonlinear = None

def generate_polynomial_features(x):
    X_poly = np.hstack([x**i for i in range(1, deg + 1)])  # Gera termos até x^i
    return X_poly

def train_pol(pol, reg, solver_alg, iter_limit, deg):
    global clf_nonlinear
    if pol == "cubic":
        clf_nonlinear = Pipeline([("poly_transform", FunctionTransformer(generate_polynomial_features, validate=False)),("linear_reg", LogisticRegression(penalty= reg, solver = solver_alg, max_iter=iter_limit))])
    elif pol == "normal":
        clf_nonlinear = Pipeline((("poly_features", PolynomialFeatures(degree=deg, include_bias=False)),("linear_reg", LogisticRegression(penalty= reg, solver = solver_alg, max_iter=iter_limit))))
    try:
        clf_nonlinear.fit(X_train, y_train)
        print("Treinamento concluído.")
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")
    print()

train_pol(pol, reg, solver_alg, iter_limit, deg)

# com limite de 100 e 1000 iterações o modelo nao converge
# com penalty = None o modelo converge em ~3 min
# com penalty = l2 o modelo converge em ~2min
# com penalty l1 o modelo converge em ~22min

# com degree = 1 e penalty none o modelo convergiu em 2 minutos, com auc de treino 0,77 e auc de teste 0,75

# com degree = 2 e penalty = none o modelo convergiu em 47 minutos, com area sob a curva = 0.6290
# com degree = 3 a memória ram do ambiente colab nao é suficiente para o treinamento

# excluindo coeficientes retangulares, com degree 3 e penalty none o modelo convergiu em 11 segundos e teve area sob a curva de 0.0024
# com degree 3 e penalty l2 foi a mesma coisa
# com degree 3 e penalty l1 (solver = saga) a auc foi 0.0026

# com degree 2 e penalty l1 (solver = saga) o modelo convergiu em 1:30 horas e a auc foi 0.0026
# com degree 2 e penalty l2 o modelo convergiu em 1:14 horas e a auc foi 0.0026
# com degree 4 e penalty none o modelo convergiu em 6s e teve auc = 0.002
# com degree 5 e penalty none o modelo convergiu em 6s e teve auc = 0.002

# a partir daqui os coeficientes estao todos zerados e o modelo apenas chuta 0 para todos os casos
# com degree 6 e penalty none o modelo convergiu em 6s e teve auc = 0.50089
# com degree 7 e penalty none o modelo convergiu em 9s e teve auc = 0.50089
# com degree 14 e penalty none o modelo convergiu em 22s e teve auc = 0.50089
# com degree 28 e penalty none o modelo convergiu em 27s e teve auc = 0.50089


# Avaliar o modelo pela AUC precision-recall
# X_train, y_train sao os dados de treino/ X_test, y_test são os dados de teste
# Obtenha as probabilidades preditas para a classe positiva
y_train_probs = clf_nonlinear.predict_proba(X_train)[:, 1]  # Probabilidade para a classe positiva

# Calcule precisão e recall para diferentes thresholds
precision, recall, _ = precision_recall_curve(y_train, y_train_probs)

# Calcule a AUC da curva Precision-Recall
train_pr_auc = auc(recall, precision)
print(f"Área sob a curva Precision-Recall no conjunto de treino: {train_pr_auc:.5f}")


# Obtenha as probabilidades preditas para a classe positiva
y_test_probs = clf_nonlinear.predict_proba(X_test)[:, 1]  # Probabilidade para a classe positiva

# Calcule precisão e recall para diferentes thresholds
precision, recall, _ = precision_recall_curve(y_test, y_test_probs)

# Calcule a AUC da curva Precision-Recall
test_pr_auc = auc(recall, precision)
print(f"Área sob a curva Precision-Recall no conjunto de teste: {test_pr_auc:.5f}")
print()


# Exibir os coeficientes encontrados
feature_names = X.columns

# Criar um dicionário para armazenar os coeficientes para cada grau
coef_dict = {f"degree_{dg}": [] for dg in range(1, deg + 1)}

# Adicionar os coeficientes por feature e grau
coef_index = 0
for feature in feature_names:
    for dg in range(1, deg + 1):
        coef_dict[f"degree_{dg}"].append(clf_nonlinear.named_steps['linear_reg'].coef_.flatten()[coef_index])
        coef_index += 1

# Adicionar os nomes das features
coef_dict["Feature"] = feature_names

# Criar o DataFrame com a estrutura desejada
coef_df = pd.DataFrame(coef_dict)
coef_df = coef_df.set_index("Feature")  # Definir a coluna "Feature" como índice

# Imprimir a tabela de coeficientes
print(coef_df)