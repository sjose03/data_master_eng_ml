#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = data_master_eng_ml
PYTHON_VERSION = 3.10
# Se preferir usar pipenv para isolar o ambiente, descomente a linha abaixo:
# PYTHON_INTERPRETER = pipenv run python
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Instala as dependências do Python (atualiza pip e instala os pacotes do requirements.txt)
.PHONY: requirements
requirements:
	@echo "Atualizando pip e instalando dependências..."
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Remove arquivos compilados e diretórios de cache
.PHONY: clean
clean:
	@echo "Limpando arquivos compilados e cache..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Verifica o código com o ruff
.PHONY: lint
lint:
	ruff check

## Formata o código usando o ruff (integra com black)
.PHONY: format
format:
	ruff format

## Cria o ambiente virtual utilizando pipenv
.PHONY: create_environment
create_environment:
	@echo "Criando ambiente virtual com pipenv..."
	pipenv --python $(PYTHON_VERSION)
	@echo ">>> Novo pipenv criado. Ative com: pipenv shell"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Sobe os containers definidos no docker-compose.yml em modo detach
DOCKER_COMPOSE_FILE ?= docker-compose.yml
.PHONY: docker-up
docker-up:
	@echo "Subindo containers com docker-compose..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) up -d

## Processa os dados (Step Data)
# Aqui, o target step_data depende de docker-up, garantindo que o ambiente Docker esteja ativo
# e também depende do arquivo de dados processados, que é gerado pelo script dataset.py.
.PHONY: step_data
step_data: docker-up data_master_eng_ml/processed_data.csv

data_master_eng_ml/processed_data.csv: requirements
	@if [ -f $@ ]; then \
	    echo "Dados já processados encontrados em $@. Pulando processamento."; \
	else \
	    echo "Processando dados e gerando $@..."; \
	    $(PYTHON_INTERPRETER) data_master_eng_ml/dataset.py && touch $@; \
	fi

## Gera as features a partir dos dados processados
.PHONY: features
features: step_data
	@echo "Gerando as features..."
	$(PYTHON_INTERPRETER) data_master_eng_ml/features.py
#################################################################################
# DOCKER COMMANDS                                                               #
#################################################################################

## Para e remove os containers, redes e volumes definidos no docker-compose.yml
.PHONY: docker-down
docker-down:
	@echo "Parando e removendo containers com docker-compose..."
	docker-compose -f $(DOCKER_COMPOSE_FILE) down

## Reinicia os containers (executa docker-down e depois docker-up)
.PHONY: docker-restart
docker-restart: docker-down docker-up

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
