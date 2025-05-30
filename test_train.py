
import sys
import os

# Adiciona o diretório raiz ao path para que o Python encontre o módulo 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.services.training_service import start_training_for_petroleo

if __name__ == "__main__":
    result = start_training_for_petroleo()
    print(result)
