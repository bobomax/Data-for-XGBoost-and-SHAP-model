install.packages("randomForest")
install.packages("gbm")
# 加载必要的包
library(skimr)         # 数据鸟瞰
library(randomForest)  # 随机森林模型
library(caret)         # 机器学习框架，包含GBDT
library(gbm)           # 梯度提升模型
library(ggplot2)       # 数据可视化
library(DataExplorer)  # 数据缺失情况分析

# 加载数据
campus <- read.csv(file.choose())

# 数据鸟瞰
skimr::skim(campus)

# 数据缺失情况
plot_missing(campus)

# 因变量分布情况
hist(campus$WE, breaks = 10)

# 划分训练集和验证集以及测试集
set.seed(20241021)
trains <- createDataPartition(y = campus$WE, p = 0.85, list = FALSE)
trains2 <- sample(trains, nrow(campus) * 0.7)
valids <- setdiff(trains, trains2)

data_train <- campus[trains2, ]
data_valid <- campus[valids, ]
data_test <- campus[-trains, ]

# 拆分后因变量分布
hist(data_train$WE, breaks = 50)
hist(data_valid$WE, breaks = 50)
hist(data_test$WE, breaks = 50)

# 因变量自变量构建公式
colnames(campus)
form_reg <- as.formula(
  paste0(
    "WE ~ ", 
    paste(colnames(data_train)[2:28], collapse = " + ")
  )
)
form_reg

# 确保没有缺失值
data_train <- na.omit(data_train)

# 训练GBDT模型
set.seed(20241021)
fit_gbdt_reg <- train(
  form_reg,
  data = data_train,
  method = "gbm",                    # 使用梯度提升决策树
  trControl = trainControl(method = "cv", number = 5, search = "random"), # 交叉验证
  tuneGrid = expand.grid(
    n.trees = 300,                   # 决策树的数量
    interaction.depth = 5,           # 每棵树的最大深度
    shrinkage = 0.05,                # 学习率
    n.minobsinnode = 10              # 每个叶节点的最小观测数
  ),
  metric = "RMSE",
  verbose = FALSE
)

# 模型概要
print(fit_gbdt_reg)

# 绘制误差与决策树数量关系图
plot(fit_gbdt_reg$finalModel, main = "Error & Trees")

# 变量重要性
varImp(fit_gbdt_reg)
plot(varImp(fit_gbdt_reg), main = "Variable Importance Plot")

# 偏依赖图
partialPlot(
  x = fit_gbdt_reg$finalModel,  # 已训练的GBDT模型
  pred.data = data_train,        # 训练数据集
  x.var = "DF"                  # 自变量名，确保用引号表示
)

# 预测
# 训练集预测结果
trainpred <- predict(fit_gbdt_reg, newdata = data_train)
# 训练集预测误差指标
defaultSummary(data.frame(obs = data_train$WE, pred = trainpred))

# 图示训练集预测结果
plot(x = data_train$WE,
     y = trainpred,
     xlab = "Actual",
     ylab = "Prediction",
     main = "GBDT Train Dataset")

# 添加回归线
trainlinmod <- lm(trainpred ~ data_train$WE)
abline(trainlinmod, col = "blue", lwd = 2.5, lty = "solid")

# 添加45度线
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")

# 添加图例
legend("topleft",
       legend = c("Model", "Base"),
       col = c("blue", "red"),
       lwd = 2.5,
       lty = c("solid", "dashed"))

# 测试集预测结果
testpred <- predict(fit_gbdt_reg, newdata = data_test)
# 测试集预测误差指标
defaultSummary(data.frame(obs = data_test$WE, pred = testpred))
5

#参数优化
library(caret)

# 设置网格搜索的参数网格
paramGrid <- expand.grid(
  n.trees = c(100, 200, 300),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.05),
  n.minobsinnode = c(10, 20)
)

# 训练GBM模型
set.seed(20241021)
fit_gbdt_reg <- train(
  form_reg,
  data = data_train,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = paramGrid,
  metric = "RMSE",
  verbose = FALSE
)

# 模型概要
print(fit_gbdt_reg)
