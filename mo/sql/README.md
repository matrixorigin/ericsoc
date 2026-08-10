# MatrixOne Loading SQL / MatrixOne 导入 SQL

These files preserve the original full-CSV loading workflow used before the main analytical tables were finalized. They are reference scripts, not the numbered analysis pipeline. The current notebooks read the validated `chicago_tnp.unified_trips_clean` and `chicago_tnp.unified_trips_h3_res9` tables.  
这些文件保留最初的全量 CSV 导入流程，属于参考脚本，不是编号分析主线。当前 Notebook 读取已验证的 `chicago_tnp.unified_trips_clean` 和 `chicago_tnp.unified_trips_h3_res9`。

- `stage.sql`: example S3 stage definition with placeholder credentials. / 使用占位凭证的 S3 stage 示例。
- `table.sql`: raw 2022 and 2023-2024 table schemas. / 2022 与 2023-2024 原始表结构。
- `load.sql`: full-file load commands. / 全量文件导入命令。
- `view.sql`: timestamp-normalized reference views. / 时间格式统一的参考视图。
- `ext.sql`: small external-table example retained for MatrixOne compatibility testing. / 为 MatrixOne 兼容性测试保留的小型 external table 示例。

Replace all placeholders before use. Never commit real cloud credentials.  
使用前必须替换占位值，不能把真实云端凭证提交到 Git。

