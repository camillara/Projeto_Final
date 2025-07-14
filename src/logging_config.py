import logging


def configurar_logging(log_file: str = "logs/execucao.log") -> None:
    """
    Configura o sistema de logging para registrar mensagens de log tanto em arquivo quanto no console.

    Parâmetros:
    ----------
    log_file : str
        Caminho do arquivo de log onde as mensagens serão salvas. Padrão: "logs/execucao.log".

    Comportamento:
    -------------
    - Define o nível de log como INFO.
    - Formata as mensagens com data, nível e conteúdo.
    - Redireciona logs para um arquivo e também para o console (stdout).
    - Sobrescreve o arquivo de log a cada execução.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
