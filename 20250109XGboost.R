install.packages("data.table")
install.packages("caret")
install.packages("xgboost")
install.packages("Matrix")
install.packages("Metrics")
install.packages("MASS")
# 加载必要的库
library(data.table)
library(caret)
library(xgboost)
library(Matrix)
library(Metrics)


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

# 保存标准化前的均值和标准差
features_mean <- colMeans(features)
features_sd <- apply(features, 2, sd)

# 标准化特征
features <- scale(features)

# 数据集划分
set.seed(20241021)
trains <- createDataPartition(y = target, p = 0.8, list = FALSE, times = 1)  # 80% 训练集
valids <- setdiff(seq_len(nrow(data)), trains)  # 20% 验证集

# 划分特征和目标
train_features <- features[trains, ]
valid_features <- features[valids, ]
train_target <- target[trains]
valid_target <- target[valids]

# 定义反转函数
reverse_transform <- function(pred) {
  (pred^2) - offset
}

reverse_transform_features <- function(features) {
  sweep(features, 2, features_sd, "*") + features_mean
}

# 设置优化后的模型参数
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.03281965,                   # 学习率
  gamma = 0,                    
  max_depth = 10,                # 控制树深度
  subsample = 0.9,              # 样本采样
  colsample_bytree = 0.7862283,       # 特征采样
  lambda = 20,                  # L2 正则化
  alpha = 5,                    # L1 正则化
  min_child_weight = 10         # 最小子节点权重
)

# 使用交叉验证找到最佳迭代次数
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 3000,
  nfold = 5,
  early_stopping_rounds = 30,
  verbose = TRUE
)

# 提取最佳迭代轮数
best_nrounds <- cv_results$best_iteration
cat("最佳迭代轮数:", best_nrounds, "\n")

# 用最佳轮数训练最终模型
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, eval = dvalid),
  verbose = 1
)

# 预测
train_predictions <- reverse_transform(predict(model, newdata = dtrain))
valid_predictions <- reverse_transform(predict(model, newdata = dvalid))

# 提前反转目标变量
train_target <- reverse_transform(train_target)
valid_target <- reverse_transform(valid_target)

# 定义性能评估函数
calculate_metrics <- function(actual, predicted) {
  rmse_value <- sqrt(mean((actual - predicted)^2))
  mae_value <- mean(abs(actual - predicted))
  r_squared <- cor(actual, predicted)^2
  list(RMSE = rmse_value, MAE = mae_value, R_squared = r_squared)
}

# 计算性能指标
train_metrics <- calculate_metrics(train_target, train_predictions)
valid_metrics <- calculate_metrics(valid_target, valid_predictions)

# 输出结果
cat("训练集的性能指标:\n")
print(train_metrics)

cat("验证集的性能指标:\n")
print(valid_metrics)





#贝叶斯参数调优
# 加载数据
# 加载必要的库
# 加载必要的库
# 加载必要的库
library(data.table)
library(caret)
library(xgboost)
library(ParBayesianOptimization)
library(Metrics)


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

# 保存标准化前的均值和标准差
features_mean <- colMeans(features)
features_sd <- apply(features, 2, sd)

# 标准化特征
features <- scale(features)

# 数据集划分
set.seed(20241021)
trains <- createDataPartition(y = target, p = 0.8, list = FALSE, times = 1)  # 80% 训练集
valids <- setdiff(seq_len(nrow(data)), trains)  # 20% 验证集

# 划分特征和目标
train_features <- features[trains, ]
valid_features <- features[valids, ]
train_target <- target[trains]
valid_target <- target[valids]

# 定义反转函数
reverse_transform <- function(pred) {
  (pred^2) - offset
}

reverse_transform_features <- function(features) {
  sweep(features, 2, features_sd, "*") + features_mean
}


# 贝叶斯优化目标函数
opt_func <- function(eta, max_depth, subsample, colsample_bytree, min_child_weight, gamma, lambda, alpha) {
  params <- list(
    eta = eta,
    max_depth = as.integer(max_depth),
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    objective = "reg:squarederror"
  )
  xgbcv <- xgb.cv(
    params = params,
    data = dtrain,
    nfold = 5,
    nrounds = 3000,
    metrics = "rmse",
    early_stopping_rounds = 50,
    verbose = 0
  )
  list(Score = -min(xgbcv$evaluation_log$test_rmse_mean), nrounds = xgbcv$best_iteration)
}

# 超参数范围
bounds <- list(
  eta = c(0.01, 0.05),                  # 学习率
  max_depth = c(6L, 10L),              # 树的深度
  subsample = c(0.6, 0.9),             # 样本采样比例
  colsample_bytree = c(0.6, 0.9),      # 特征采样比例
  min_child_weight = c(10, 30),         # 最小叶节点样本权重
  gamma = c(0, 5),                     # 分裂节点的最小损失
  lambda = c(10, 20),                   # L2 正则化
  alpha = c(5, 15)                     # L1 正则化
)

# 贝叶斯优化
set.seed(20241021)
bayes_opt <- bayesOpt(
  FUN = opt_func,
  bounds = bounds,
  initPoints = 10,  # 初始采样点
  iters.n = 60,     # 优化迭代次数
  verbose = 2
)

# 提取最佳参数
best_paramsxgboost <- getBestPars(bayes_opt)
cat("最佳参数:\n")
print(best_paramsxgboost)

# 将最佳参数转换为数据框
best_params_df <- as.data.frame(t(best_paramsxgboost))

# 保存为 CSV 文件
write.csv(best_params_df, file = "best_paramsxgboost.csv", row.names = FALSE)

cat("最佳参数已保存到 'best_paramsxgboost.csv'\n")

