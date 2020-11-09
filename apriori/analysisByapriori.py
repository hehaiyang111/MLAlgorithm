import pandas as pd
from apriori import *


# inputFile
inputFile = './menu_orders.xls'
outputFile = './apriori_rules.xls' # 结果

# 读取数据
data = pd.read_excel(inputFile,header=None)
# 把不为空的数据设置为1
ct = lambda x : pd.Series(1, index=x[pd.notnull(x)])
b = map(ct,data.as_matrix())
# 实现矩阵转换 空值用0补充
data = pd.DataFrame(list(b)).fillna(0)

del b  # 删除中间变量

# 设置最小阈值（支持度、置信度）
support = 0.2
confidence = 0.5
ms='---'


find_rule(data,support,confidence,ms).to_excel(outputFile)
