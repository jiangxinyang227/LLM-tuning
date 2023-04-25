import logging


def get_logger(name, log_path):
    """
    get loggger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    # 向文件输出
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 向终端输出
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger