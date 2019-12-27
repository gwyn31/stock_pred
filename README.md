股票预测系统

运行方法:
1. 首先运行process_tweets.py从tweets文本中提取特征;
2. 确认/编辑model_configs.py, 配置模型与运行选项;
3. 运行main.py(如果在IDE中出现问题请打开cmd并进入文件所在目录, 使用python main.py运行(环境变量中有python的话))

依赖项/包:
```
python >= 3.5.2
numpy>=1.15.2
scipy>=1.2.0
pandas>=0.23.0
statsmodels>=0.10.0
scikit-learn>=0.19.0
matplotlib
```

如果已经安装了正确版本的python和pip, 并且它们均在环境变量中, 可以在cmd中使用以下命令安装所有依赖项/包(在项目根目录下):
```
pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
