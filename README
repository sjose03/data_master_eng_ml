# MLOps Pipeline com CI/CD usando GitHub Actions

Este projeto demonstra a implementação de um pipeline de MLOps para um problema de regressão utilizando XGBoost. O pipeline integra ferramentas de tracking de experimentos (CometML), monitoramento de dados (Evidently), e banco de dados (MongoDB Atlas). O pipeline de CI/CD é orquestrado pelo GitHub Actions.

## Tecnologias Usadas

<p align="center">
  <img src="images/github_actions_logo.png" alt="GitHub Actions" width="100"/>
  <img src="images/cdnlogo.com_fastapi.svg" alt="FastAPI" width="100"/>
  <img src="images/xgboost_logo.png" alt="XGBoost" width="100"/>
  <img src="images/cometml_logo.png" alt="CometML" width="100"/>
  <img src="images/evidently_logo.png" alt="Evidently" width="100"/>
  <img src="images/mongodb_atlas_logo.png" alt="MongoDB Atlas" width="100"/>
  <img src="images/python_logo.png" alt="Python" width="100"/>
</p>


## Estrutura de Diretórios

```plaintext
project-root/
├── .github/
│   └── workflows/
│       ├── checkout.yml
│       ├── setup.yml
│       ├── install_dependencies.yml
│       ├── run_tests.yml
│       └── deploy.yml
├── data/
│   ├── featurization.py
│   ├── insert_data.py
│   └── test_data_processing.py
├── models/
│   ├── train.py
│   └── deploy.py
├── tests/
│   ├── unit.py
│   ├── integration.py
│   └── test_api.py
├── api/
│   └── main.py
├── frontend/
│   └── app.py
├── images/
│   ├── github_actions_logo.png
│   ├── fastapi_logo.png
│   ├── xgboost_logo.png
│   ├── cometml_logo.png
│   ├── evidently_logo.png
│   ├── mongodb_atlas_logo.png
│   ├── python_logo.png
│   ├── pipeline_diagram.png
│   └── cicd_diagram.png
└── requirements.txt
```
## Características Técnicas

### 1. Ingestão de Dados

Os dados são carregados a partir do MongoDB Atlas. O script `insert_data.py` insere dados do conjunto de dados público `California Housing` no MongoDB.

### 2. Featurização

O script `featurization.py` processa os dados carregados, criando novas variáveis derivadas e realizando a normalização dos dados.

### 3. Treinamento do Modelo

O script `train.py` realiza o treinamento de um modelo de regressão usando XGBoost. As métricas do modelo são logadas no CometML.

### 4. Monitoramento

O monitoramento de dados é realizado utilizando a ferramenta Evidently para detectar drifts nos dados.

### 5. CI/CD com GitHub Actions

O pipeline CI/CD é dividido em várias etapas:
- `checkout.yml`: Faz o checkout do código.
- `setup.yml`: Configura o ambiente Python.
- `install_dependencies.yml`: Instala as dependências.
- `run_tests.yml`: Executa testes unitários e de integração.
- `deploy.yml`: Implanta o modelo.

### 6. Servindo o Modelo

O script `main.py` fornece um endpoint API para previsões online e em batch, utilizando autenticação JWT.

## Instruções para Uso

### 1. Configuração do Ambiente

1. Clone o repositório:
    ```bash
    git clone https://github.com/your-repo/mlops-pipeline.git
    cd mlops-pipeline
    ```

2. Crie e ative um ambiente virtual Python (recomendado):
    ```bash
    python -m venv env
    source env/bin/activate  # Para Windows use `env\Scripts\activate`
    ```

3. Instale as dependências:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. Configure as variáveis de ambiente para CometML e MongoDB Atlas:
    - Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
      ```
      COMET_API_KEY=your-comet-api-key
      MONGODB_URI=your-mongodb-atlas-uri
      DATABASE_NAME=your-database-name
      COLLECTION_NAME=your-collection-name
      SECRET_KEY=your-secret-key
      ```

### 2. Ingestão de Dados

Execute o script para inserir dados no MongoDB Atlas:
```bash
python data/insert_data.py
```
### 3. Treinamento do Modelo
Execute o script de treinamento:

```bash
python models/train.py
```

### 4. Servir o Modelo
Execute o servidor API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. CI/CD com GitHub Actions
Os workflows de CI/CD serão automaticamente acionados em cada push ou pull request.


### 6. Diagrama do Pipeline


![Diagrama do Pipeline](images/pipeline_diagram.png)


## Possíveis Melhorias

1. **Automatizar a criação e configuração do ambiente**:
   - Utilizar ferramentas como Docker para garantir que o ambiente seja replicável.

2. **Aprimorar a Featurização**:
   - Adicionar técnicas mais avançadas de engenharia de features.
   - Implementar seleção automática de features.

3. **Melhorar a Monitoria**:
   - Implementar alertas automáticos quando houver drift nos dados.
   - Integrar mais métricas de monitoramento.

4. **Expandir Testes**:
   - Adicionar testes de carga e desempenho para o endpoint API.
   - Implementar testes de validação de dados.

5. **Aprimorar a Segurança**:
   - Implementar autenticação e autorização mais robustas para a API.
   - Garantir a conformidade com as melhores práticas de segurança.