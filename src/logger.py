import logging
import os
import sys

def setup_logger(name, log_file, level=logging.INFO):
    """
    Thiết lập logger với tên và file log được chỉ định.
    
    Args:
        name (str): Tên của logger (thường là __name__ của module).
        log_file (str): Đường dẫn đến file log.
        level: Cấp độ logging (mặc định là INFO).
    
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler cho file log (UTF-8)
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler cho console với mã hóa UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setStream(sys.stdout)
    
    # Tạo logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Xóa các handler cũ để tránh trùng lặp
    logger.handlers.clear()
    
    # Thêm các handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger