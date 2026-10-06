# quadtree

在一个固定方形区域上实现四叉树空间索引：叶子超过容量就分裂、删点变少后向上合并，
支持范围查询、点定位、重复点计数与深度上限，纯标准库 Python 3。

## 目录

- quadtree/core.py 四叉树内核
- tests/test_core.py 行为测试

## 运行测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v

只使用 Python 标准库，不需要安装任何依赖。
