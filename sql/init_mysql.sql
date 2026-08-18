CREATE DATABASE IF NOT EXISTS data_agent
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE data_agent;

DROP TABLE IF EXISTS meta_table_ext;
DROP TABLE IF EXISTS meta_table;

CREATE TABLE meta_table (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  catalog_name VARCHAR(128) DEFAULT NULL,
  db_name VARCHAR(128) NOT NULL,
  table_name VARCHAR(128) NOT NULL,
  full_table_name VARCHAR(384) NOT NULL,
  table_comment VARCHAR(512) DEFAULT NULL,
  biz_line VARCHAR(128) DEFAULT NULL,
  domain VARCHAR(64) DEFAULT NULL,
  data_layer VARCHAR(32) DEFAULT NULL,
  owner VARCHAR(128) DEFAULT NULL,
  lifecycle_status VARCHAR(32) DEFAULT 'online',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_full_table_name (full_table_name),
  KEY idx_table_name (table_name),
  KEY idx_db_table (db_name, table_name),
  KEY idx_domain_layer (domain, data_layer),
  KEY idx_biz_domain_layer (biz_line, domain, data_layer)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE meta_table_ext (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  table_id BIGINT NOT NULL,
  term_type VARCHAR(64) NOT NULL,
  term_value VARCHAR(256) NOT NULL,
  normalized_term VARCHAR(256) DEFAULT NULL,
  confidence DECIMAL(5,4) DEFAULT 1.0000,
  source VARCHAR(128) DEFAULT 'manual',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uk_table_term (table_id, term_type, term_value),
  KEY idx_term_value (term_value),
  KEY idx_normalized_term (normalized_term),
  CONSTRAINT fk_meta_table_ext_table_id
    FOREIGN KEY (table_id) REFERENCES meta_table(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO meta_table
(catalog_name, db_name, table_name, full_table_name, table_comment, biz_line, domain, data_layer, owner)
VALUES
(NULL, 'dwd', 'orderInfo', 'dwd.orderInfo', '订单明细事实表', '安逸花', '交易域', 'DWD', 'data_team'),
(NULL, 'dwd', 'order_info', 'dwd.order_info', '订单信息宽表', '安逸花', '交易域', 'DWD', 'data_team'),
(NULL, 'dwd', 'payment_detail', 'dwd.payment_detail', '支付明细表', '安逸花', '营销域', 'DWD', 'data_team'),
(NULL, 'dwd', 'pay_record', 'dwd.pay_record', '支付流水记录表', '安逸花', '营销域', 'DWD', 'data_team'),
(NULL, 'dwd', 'userInfo', 'dwd.userInfo', '用户明细表', '安逸花', '营销域', 'DWD', 'data_team'),
(NULL, 'dim', 'userInfo', 'dim.userInfo', '用户维表', '安逸花', '用户域', 'DIM', 'data_team');

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '订单信息表', 'order_info', 0.9000, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.orderInfo';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '订单明细表', 'order_info', 0.9000, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.orderInfo';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '订单信息表', 'order_info', 0.8500, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.order_info';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '支付明细表', 'payment_detail', 0.9000, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.payment_detail';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '支付流水表', 'payment_detail', 0.8800, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.pay_record';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '用户信息表', 'user_info', 0.8800, 'manual'
FROM meta_table WHERE full_table_name = 'dwd.userInfo';

INSERT INTO meta_table_ext
(table_id, term_type, term_value, normalized_term, confidence, source)
SELECT id, 'table_term', '用户信息表', 'user_info', 0.8200, 'manual'
FROM meta_table WHERE full_table_name = 'dim.userInfo';
