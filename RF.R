# 加载必要的包
library(caret)
library(ggplot2)
library(DataExplorer)

# 加载数据
campus <- read.csv(file.choose())

# 数据鸟瞰
skim(campus)

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

# 确保没有缺失值
data_train <- na.omit(data_train)

# 使用一组随机参数训练随机森林模型
set.seed(20241021)
fit_rf_reg_initial <- train(
  WE ~ .,
  data = data_train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  metric = "RMSE",
  verbose = FALSE,
  tuneLength = 1  # 使用默认参数
)

# 模型概要
print(fit_rf_reg_initial)

#参数调优
library(randomForest)

# 自定义随机森林模型，调优 nodesize
set.seed(20241021)
fit_rf_manual <- randomForest(
  WE ~ .,
  data = data_train,
  mtry = 4,            # 调优后的变量数
  nodesize = 5,        # 最小节点样本数
  ntree = 500          # 树的数量
)

# 输出模型结果
print(fit_rf_manual)
# 定义调优参数网格，仅包含 mtry
paramGrid <- expand.grid(
  mtry = c(2, 4, 6, 8)  # 尝试的变量数
)

# 训练随机森林模型
set.seed(20241021)
fit_rf_reg_tuned <- train(
  WE ~ .,
  data = data_train,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5),
  tuneGrid = paramGrid,
  metric = "RMSE",
  verbose = FALSE
)

# 输出调优后的模型结果
print(fit_rf_reg_tuned)


# 绘制误差与决策树数量关系图
plot(fit_rf_reg_tuned)

# 变量重要性
varImp(fit_rf_reg_tuned)
plot(varImp(fit_rf_reg_tuned), main = "Variable Importance Plot")

# 预测
# 训练集预测结果
trainpred <- predict(fit_rf_reg_tuned, newdata = data_train)
# 训练集预测误差指标
defaultSummary(data.frame(obs = data_train$WE, pred = trainpred))

# 图示训练集预测结果
plot(x = data_train$WE,
     y = trainpred,
     xlab = "Actual",
     ylab = "Prediction",
     main = "Random Forest Train Dataset")

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
testpred <- predict(fit_rf_reg_tuned, newdata = data_test)
# 测试集预测误差指标
defaultSummary(data.frame(obs = data_test$WE, pred = testpred))
