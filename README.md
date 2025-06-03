# LSTM Stock Prediction API

## Descrição
Esta aplicação fornece uma API para predição de preços de ações utilizando modelos de aprendizado de máquina baseados em LSTM (Long Short-Term Memory). Além disso, a aplicação possui um processo agendado que realiza tarefas diárias, como predições automáticas e verificação do status dos modelos. Este projeto foi desenvolvido para o curso de Machine Learning Engineering como parte do Tech Challenge 4 da FIAP.

## Variáveis de Ambiente
As seguintes variáveis de ambiente são utilizadas no projeto:
- `SEED`: Define a seed para reprodutibilidade dos resultados (padrão: 42).
- `ENABLE_LOGS`: Ativa ou desativa os logs da aplicação (padrão: True).
- `DAILY_TASK_HOUR`: Define o horário em que o processo agendado será executado diariamente (padrão: 3).

## Funcionalidades

### Endpoints da API

#### 1. `/models/check` (POST)
- **Descrição**: Verifica o status dos modelos para os símbolos de ações fornecidos. Caso um modelo não exista, o treinamento será iniciado automaticamente.
- **Parâmetros**:
  ```json
  {
    "symbols": ["BBAS3.SA", "PETR4.SA"]
  }
  ```
- **Resposta**:
  ```json
  [
    {
      "symbol": "BBAS3.SA",
      "model_exists": true,
      "message": "Model and scaler are available."
    }
  ]
  ```

#### 2. `/models/metrics` (POST)
- **Descrição**: Retorna métricas de performance para os modelos dos símbolos fornecidos.
- **Parâmetros**:
  ```json
  {
    "symbols": ["BBAS3.SA", "PETR4.SA"]
  }
  ```
- **Resposta**:
  ```json
  [
    {
      "symbol": "BBAS3.SA",
      "metrics": {
        "MAE": 0.123,
        "RMSE": 0.456,
        "MAPE": 1.23,
        "R²": 0.89,
        "last_data_date": "2025-06-01",
        "training_duration": 120,
        "training_data_size": 1000,
        "validation_data_size": 200
      }
    }
  ]
  ```

#### 3. `/models/predict` (POST)
- **Descrição**: Realiza predições de preços para os símbolos fornecidos. Caso um modelo não exista, o treinamento será iniciado automaticamente.
- **Parâmetros**:
  ```json
  {
    "symbols": ["BBAS3.SA", "PETR4.SA"]
  }
  ```
- **Resposta**:
  ```json
  {
    "predictions": {
      "BBAS3.SA": {
        "predicted_price": 16.04,
        "status": "success",
        "variation": {
          "status": "negative",
          "percentage": -0.98
        }
      }
    }
  }
  ```

#### 4. `/models/retrain` (POST)
- **Descrição**: Força o re-treino de um modelo para o símbolo fornecido.
- **Parâmetros**:
  ```json
  {
    "symbol": "BBAS3.SA"
  }
  ```
- **Resposta**:
  ```json
  {
    "message": "Re-training initiated for model: BBAS3.SA"
  }
  ```

### Processo Agendado
A aplicação possui um processo agendado que executa as seguintes tarefas diariamente:
1. **Predições Automáticas**: Realiza predições para todos os símbolos disponíveis.
2. **Verificação de Modelos**: Verifica se os modelos estão atualizados. Caso um modelo esteja desatualizado (mais de 15 dias), o re-treino é iniciado automaticamente.

O agendamento é configurado para rodar em um horário específico, definido pela variável de ambiente `DAILY_TASK_HOUR`. Além disso, o processo agendado também é executado automaticamente quando o servidor é iniciado.

## Como Executar

### Pré-requisitos
- Python 3.11 ou superior
- Dependências listadas no arquivo `requirements.txt`

### Passos
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Inicie o servidor Flask:
   ```bash
   python run.py
   ```

3. Acesse a documentação Swagger para explorar os endpoints:
   ```
   http://localhost:5000/swagger
   ```

## Notebooks
O projeto inclui notebooks para auxiliar no treinamento e análise dos modelos. Eles estão localizados na pasta `notebooks/`.

- `notebook_modelo.ipynb`: Notebook principal utilizado para o treinamento dos modelos LSTM.

## Estrutura do Projeto
- `app/`: Contém os módulos principais da aplicação.
  - `routes/`: Define os endpoints da API.
  - `services/`: Contém os serviços para predição, treinamento e tarefas diárias.
  - `ml_models/`: Gerencia os modelos de aprendizado de máquina e seus artefatos.
  - `utils/`: Utilitários como logger e rastreador de status.
- `logs/`: Armazena os logs da aplicação.
- `scripts/`: Scripts auxiliares para treinamento e manutenção.
- `notebooks/`: Contém notebooks para auxiliar no treinamento e análise dos modelos.

## Logs
Os logs da aplicação são armazenados no diretório `logs/` no arquivo `app.log`. Eles incluem informações sobre predições, treinamentos e erros.

## Melhorias Futuras

### 1. Avaliação e Re-treino Baseado na Perda de Desempenho
- Implementar nas tarefas diárias uma avaliação contínua do desempenho dos modelos utilizando dados reais.
- Caso seja detectada uma perda significativa de desempenho, iniciar automaticamente o re-treino do modelo correspondente.

### 2. Criação de Fila para Treinamento de Modelos
- Desenvolver um sistema de fila para gerenciar os treinamentos de modelos de forma assíncrona.
- Garantir que o processamento assíncrono não sobrecarregue os recursos computacionais disponíveis, priorizando tarefas críticas.

## Autor
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/victor-hugo-teles-de-santana-359ba260/) Victor H T Santana.

