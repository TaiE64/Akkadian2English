# Project Guidelines

## Submission Workflow
- **必须本地验证确认方案有改进后，再生成新的 notebook 提交到 Kaggle**
- 使用 `test_configs.py` 或类似脚本对比新旧配置的 GeoMean
- 只有新方案的 GeoMean 高于当前最佳时才生成 notebook 并上传

## User Preferences
- 不要自动运行训练，告诉用户让他们自己运行
- 使用 conda 环境 `kaggle_deep_past`，不要污染系统 Python
- 用中文交流
