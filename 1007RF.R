# 加载必要的库
library(data.table)
library(randomForest)
library(caret)

# 加载数据
data <- fread(file.choose())  # 替换为实际文件路径

# 数据准备
features <- data[, 2:28, with = FALSE]  # 自变量
target <- data[[29]]                    # 因变量 WE

# 检查是否有分类变量并进行哑编码
categorical_columns <- which(sapply(features, is.character) | sapply(features, is.factor))
if (length(categorical_columns) > 0) {
  dummy_model <- dummyVars(~ ., data = features, fullRank = TRUE)
  features <- as.data.table(predict(dummy_model, newdata = features))
}

# 平方根变换
if (any(target < 0)) {
  offset <- abs(min(target))  # 偏移量，确保目标变量为非负
  target <- target + offset
} else {
  offset <- 0
}
target <- sqrt(target)

# 标准化特征
features <- scale(features)

# 数据集划分
set.seed(20241021)
trains <- createDataPartition(target, p = 0.8, list = FALSE)  # 80% 训练集
test_indices <- setdiff(seq_len(nrow(data)), trains)  # 剩余 20% 测试集

# 划分特征和目标
train_features <- features[trains, ]
test_features <- features[test_indices, ]
train_target <- target[trains]
test_target <- target[test_indices]

# 使用最佳参数训练随机森林模型
rf_model <- randomForest(
  x = as.data.frame(train_features),
  y = train_target,
  ntree = 3000,  # 决策树数量
  mtry = 12,  # 贝叶斯优化后的 mtry
  nodesize = 5,  # 贝叶斯优化后的 nodesize
  sampsize = as.integer(nrow(train_features) * 0.8792317),  # 贝叶斯优化后的采样比例
  importance = TRUE  # 开启变量重要性
)

# 在训练集和测试集上进行预测并反转平方根
train_predictions <- predict(rf_model, newdata = as.data.frame(train_features))^2
test_predictions <- predict(rf_model, newdata = as.data.frame(test_features))^2

# 反转目标变量
train_target_original <- train_target^2
test_target_original <- test_target^2

# 评估函数
calculate_metrics <- function(actual, predicted) {
  rmse_value <- sqrt(mean((actual - predicted)^2))
  mae_value <- mean(abs(actual - predicted))
  r_squared <- cor(actual, predicted)^2
  list(RMSE = rmse_value, MAE = mae_value, R_squared = r_squared)
}

# 计算性能指标
train_metrics <- calculate_metrics(train_target_original, train_predictions)
test_metrics <- calculate_metrics(test_target_original, test_predictions)

# 输出性能指标
cat("训练集的性能指标:\n")
print(train_metrics)

cat("测试集的性能指标:\n")
print(test_metrics)


# 贝叶斯优化目标函数
opt_func <- function(mtry, nodesize, sampsize) {
  rf_model <- randomForest(
    x = as.data.frame(train_features),
    y = train_target,
    ntree = 3000,
    mtry = 12,
    nodesize = 5,
    sampsize = 0.8792317,
    importance = FALSE
  )
  
  # 测试集预测
  test_predictions <- predict(rf_model, newdata = as.data.frame(test_features))
  rmse_value <- sqrt(mean((test_target - test_predictions)^2))
  
  # 返回 RMSE
  list(Score = -rmse_value)
}
# 贝叶斯优化目标函数
opt_func <- function(mtry, nodesize, sampsize) {
  rf_model <- randomForest(
    x = as.data.frame(train_features),
    y = train_target,
    ntree = 3000,
    mtry = as.integer(mtry),
    nodesize = as.integer(nodesize),
    sampsize = as.integer(nrow(train_features) * sampsize),
    importance = FALSE
  )
  
  # 测试集预测
  test_predictions <- predict(rf_model, newdata = as.data.frame(test_features))
  rmse_value <- sqrt(mean((test_target - test_predictions)^2))
  
  # 返回 RMSE
  list(Score = -rmse_value)
}

# 超参数范围
bounds <- list(
  mtry = c(6L, 12L),               # 每次随机选择的特征数
  nodesize = c(5L, 30L),           # 叶节点最小样本数
  sampsize = c(0.6, 0.9)           # 采样比例
)

# 贝叶斯优化
set.seed(20241021)
bayes_opt <- bayesOpt(
  FUN = opt_func,
  bounds = bounds,
  initPoints = 10,
  iters.n = 60,
  verbose = 2
)

# 提取最佳参数
best_params <- getBestPars(bayes_opt)
cat("最佳参数:\n")
print(best_params)

# 保存最佳参数为 CSV 文件
best_params_df <- as.data.frame(t(best_params))
write.csv(best_params_df, "best_rf_params.csv", row.names = FALSE)
cat("最佳参数已保存为 best_rf_params.csv\n")


