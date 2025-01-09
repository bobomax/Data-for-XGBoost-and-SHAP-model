# 加载必要的库
library(data.table)
library(caret)
library(gbm)  # GBDT 模型的 R 包

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
train_indices <- sample(trains, size = round(0.85 * length(trains)))  # 85% 用于训练
valid_indices <- setdiff(trains, train_indices)  # 剩余部分为验证集
test_indices <- setdiff(seq_len(nrow(data)), trains)  # 20% 测试集

# 划分特征和目标
train_features <- features[train_indices, ]
valid_features <- features[valid_indices, ]
test_features <- features[test_indices, ]
train_target <- target[train_indices]
valid_target <- target[valid_indices]
test_target <- target[test_indices]

# 定义反转函数
reverse_transform <- function(pred) {
  (pred^2) - offset
}

# 参数设置
gbdt_params <- list(
  n.trees = 3000,              # 树的数量
  interaction.depth = 11,      # 与 XGBoost 的 max_depth 保持一致
  shrinkage = 0.0461657,       # 学习率与 XGBoost 的 eta 相似
  n.minobsinnode = 11,         # 最小子节点样本数
  bag.fraction = 1,            # 与 XGBoost 的 subsample 相匹配
  train.fraction = 0.85        # 与 XGBoost 的训练集比例保持一致
)

# 模型训练
gbdt_model <- gbm(
  formula = train_target ~ .,
  data = as.data.frame(train_features),
  distribution = "gaussian",  # 用于回归任务
  n.trees = gbdt_params$n.trees,
  interaction.depth = gbdt_params$interaction.depth,
  shrinkage = gbdt_params$shrinkage,
  n.minobsinnode = gbdt_params$n.minobsinnode,
  bag.fraction = gbdt_params$bag.fraction,
  train.fraction = gbdt_params$train.fraction,
  cv.folds = 5,  # 启用5折交叉验证
  verbose = TRUE
)

# 选择最佳树数量
best_iter <- gbm.perf(gbdt_model, method = "cv", plot.it = FALSE)

# 预测并反转
train_predictions <- reverse_transform(predict(gbdt_model, newdata = as.data.frame(train_features), n.trees = best_iter))
valid_predictions <- reverse_transform(predict(gbdt_model, newdata = as.data.frame(valid_features), n.trees = best_iter))
test_predictions <- reverse_transform(predict(gbdt_model, newdata = as.data.frame(test_features), n.trees = best_iter))

# 评估性能指标
calculate_metrics <- function(actual, predicted) {
  rmse_value <- sqrt(mean((actual - predicted)^2))
  mae_value <- mean(abs(actual - predicted))
  r_squared <- cor(actual, predicted)^2
  list(RMSE = rmse_value, MAE = mae_value, R_squared = r_squared)
}

train_metrics <- calculate_metrics(reverse_transform(train_target), train_predictions)
valid_metrics <- calculate_metrics(reverse_transform(valid_target), valid_predictions)
test_metrics <- calculate_metrics(reverse_transform(test_target), test_predictions)

# 输出结果
cat("训练集的性能指标:\n")
print(train_metrics)

cat("验证集的性能指标:\n")
print(valid_metrics)

cat("测试集的性能指标:\n")
print(test_metrics)



##贝叶斯参数调优
# 加载必要的库
# 加载必要的库
# 加载必要的库
library(data.table)
library(caret)
library(gbm)
library(ParBayesianOptimization)

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
train_indices <- sample(trains, size = round(0.85 * length(trains)))  # 85% 用于训练
valid_indices <- setdiff(trains, train_indices)  # 剩余部分为验证集
test_indices <- setdiff(seq_len(nrow(data)), trains)  # 20% 测试集

# 划分特征和目标
train_features <- features[train_indices, ]
valid_features <- features[valid_indices, ]
train_target <- target[train_indices]
valid_target <- target[valid_indices]

# 定义反转函数
reverse_transform <- function(pred) {
  (pred^2) - offset
}

# 贝叶斯优化目标函数
opt_func <- function(interaction.depth, shrinkage, n.minobsinnode, bag.fraction) {
  gbdt_model <- gbm(
    formula = train_target ~ .,
    data = as.data.frame(train_features),
    distribution = "gaussian",
    n.trees = 3000,
    interaction.depth = as.integer(interaction.depth),
    shrinkage = shrinkage,
    n.minobsinnode = as.integer(n.minobsinnode),
    bag.fraction = bag.fraction,
    train.fraction = 0.85,
    cv.folds = 5,  # 启用交叉验证
    verbose = FALSE
  )
  
  # 使用交叉验证获取最佳迭代次数
  best_iter <- gbm.perf(gbdt_model, method = "cv", plot.it = FALSE)
  
  # 验证集预测
  valid_predictions <- predict(gbdt_model, newdata = as.data.frame(valid_features), n.trees = best_iter)
  
  # 计算 RMSE
  valid_predictions <- reverse_transform(valid_predictions)
  valid_actual <- reverse_transform(valid_target)
  rmse_value <- sqrt(mean((valid_actual - valid_predictions)^2))
  
  # 返回目标值（仅返回 Score）
  list(Score = -rmse_value)
}

# 超参数范围
bounds <- list(
  interaction.depth = c(6L, 12L),
  shrinkage = c(0.001, 0.1),
  n.minobsinnode = c(5L, 30L),
  bag.fraction = c(0.6, 1.0)
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

# 输出最佳参数
best_params <- getBestPars(bayes_opt)
cat("最佳参数:\n")
print(best_params)
