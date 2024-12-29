#安装包
install.packages("xgboost")
install.packages("tidyverse")
install.packages("skimr")
install.packages( "DataExplorer")
install.packages("caret")
install.packages("pROC")
install.packages("shapviz")
install.packages("iml")
install.packages("shapr")
install.packages("GGally")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("plotly")
#加载
library(xgboost)
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(shapviz)
library(pROC)
library(iml)
library(shapr)
library(GGally)
library(reshape2)
library(ggplot2)
library(dplyr)
library(plotly)

###
#加载数据
campus<- read.csv(file.choose())

# 检查每列的统计信息
for(col in names(campus)) {
  if(is.numeric(campus[[col]])) {  # 检查是否是数值型数据
    cat("\n列名:", col, "\n")
    cat("最大值:", max(campus[[col]], na.rm = TRUE), "\n")
    cat("最小值:", min(campus[[col]], na.rm = TRUE), "\n")
    cat("平均值:", mean(campus[[col]], na.rm = TRUE), "\n")
    
    # 检查是否所有值相同或列为空（标准差为0）
    if(length(unique(campus[[col]])) > 1) {
      cat("标准偏差:", sd(campus[[col]], na.rm = TRUE), "\n")
      cat("方差:", var(campus[[col]], na.rm = TRUE), "\n") 
    } else {
      cat("标准偏差和方差无法计算，所有值相同或数据为空。\n")
    }
  } else {
    cat("\n列名:", col, "不是数值型数据，跳过。\n")
  }
}

#数据鸟瞰
skimr::skim(campus)

#数据缺失情况
plot_missing(campus)

#因变量分布情况
hist(campus$WE,breaks=10)



####
#拆分数据
set.seed(20241021)
trains<-createDataPartition(y=campus$WE,p=0.85,list = F,times = 1)
trains2<-sample(trains,nrow(campus)*0.7)
valids<-setdiff(trains,trains2)

data_train<-campus[trains2,]
data_valid<-campus[valids,]
data_test<-campus[-trains,]

#拆分后因变量分布
hist(data_train$WE,breaks=50)
hist(data_valid$WE,breaks=50)
hist(data_test$WE,breaks=50)

# 数据准备
colnames(campus)


# 假设自变量是前 2:28 列
dvfunc <- dummyVars(~., data = data_train[, c(2:28)], fullRank = TRUE)

# 对训练集进行哑编码
data_trainx <- predict(dvfunc, newdata = data_train[, c(2:28)])
data_trainy <- data_train$WE

# 对验证集进行哑编码
data_validx <- predict(dvfunc, newdata = data_valid[, c(2:28)])
data_validy <- data_valid$WE

# 对测试集进行哑编码
data_testx <- predict(dvfunc, newdata = data_test[, c(2:28)])
data_testy <- data_test$WE

# 构建 DMatrix 对象
dtrain <- xgb.DMatrix(data = data_trainx, label = data_trainy)
dvalid <- xgb.DMatrix(data = data_validx, label = data_validy)
dtest <- xgb.DMatrix(data = data_testx, label = data_testy)

# 监视列表
watchlist <- list(train = dtrain, test = dvalid)



###
#训练模型
fit_xgb_reg <- xgb.train(
  data = dtrain,
  eta = 0.05,                      # 降低学习率，减少过拟合
  gamma = 0.1,                     # 增加 gamma 以惩罚复杂模型
  max_depth = 6,                   # 减少 max_depth，控制模型复杂性
  subsample = 0.8,                 # 随机抽样以增加泛化能力
  colsample_bytree = 0.8,          # 控制列抽样比例
  lambda = 1,                      # 增加 L2 正则化
  alpha = 1,                       # 增加 L1 正则化
  min_child_weight = 5,            # 增加最小叶子节点权重
  objective = "reg:squarederror",  # 均方误差损失
  nrounds = 2000,                  # 训练轮次
  watchlist = watchlist,
  early_stopping_rounds = 100,     # 提前停止条件
  print_every_n = 50               # 每 50 次迭代打印一次结果
)

#模型概要
fit_xgb_reg


####
#预测
#训练集预测结果
trainpred<-predict(fit_xgb_reg,
                   newdata = dtrain)

#训练集预测误差指标
defaultSummary(data.frame(obs=data_train$WE,
                          pred=trainpred))



#测试集预测结果
testpred<-predict(fit_xgb_reg,
                  newdata = dtest)
#测试集预测误差指标
defaultSummary(data.frame(obs=data_test$WE,
                          pred=testpred))

#图示测试集预测结果
plot(x = data_test$WE,
     y = testpred,
     xlab = "actual",
     ylab = "prediction",
     main = "XGboost Test Dataset")
# Add regression line
testlinmod <- lm(testpred ~ data_test$WE)
abline(testlinmod, col = "blue", lwd = 2.5, lty = "solid")
# Add 45-degree line
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")
# Add legend
legend("topleft",
       legend = c("model", "base"),
       col = c("blue", "red"),
       lwd = 2.5,
       lty = c("solid", "dashed"))



###
###
#超参数调优
#网格搜索（Grid Search）
# 定义参数网格
tune_grid <- expand.grid(
  nrounds = c(500, 1000,1500),
  max_depth = c(3, 4,5,6),
  eta = c(0.1, 0.3, 0.5),
  gamma = c(0,0.001, 0.01, 0.1),
  colsample_bytree = c(0.3, 0.4, 0.5,0.7),
  subsample = c(0.5, 0.7, 1),
  min_child_weight = c(1, 3, 5)
)

# 设置训练控制
train_control <- trainControl(
  method = "cv",          # K折交叉验证
  number = 4,             # 5折交叉验证
  verboseIter = TRUE,
  allowParallel = TRUE,
  savePredictions = "final"  # 保存最终的预测
)

# 使用网格搜索调优XGBoost模型
xgb_tune <- train(
  x = data_validx,
  y = data_validy,  # 假设标签列为 "Y1"
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tune_grid,
)

# 输出最佳参数
print(xgb_tune$bestTune)



#计算shap值
shap_xgboost <- shapviz(fit_xgb_reg, X_pred = as.matrix(data_trainx[,1:27]))
# 确保数据是矩阵
data_trainx_matrix <- as.matrix(data_trainx)
# 绘制 SHAP 摘要图
xgb.plot.shap.summary(data = data_trainx_matrix, model = fit_xgb_reg)


#SHAP值

###
shap_values <- shap_xgboost$shap_values  # 获取所有 SHAP 值

# 确保数据是矩阵
data_trainx_matrix <- as.matrix(data_trainx)

# 创建 shapviz 对象，使用训练好的 XGBoost 模型和特征矩阵
shap_xgboost <- shapviz(fit_xgb_reg, X_pred = data_trainx_matrix)

# 绘制 SHAP 变量重要性蜂群图
sv_importance(shap_xgboost, kind = "beeswarm", max_display = 27)
###
#SHAP Summary Plot (汇总图)
shapviz::sv_importance(shap_xgboost)
print(sv_importance(shap_xgboost))
# 绘制 SHAP 汇总图
sv_importance(shap_xgboost,max_display = 27)

###
#变量重要性图
# 确认 shap_xgboost 是 shapviz 对象
if (inherits(shap_xgboost, "shapviz")) {
  shap_values <- shap_xgboost$S  # 从 shapviz 对象中提取 SHAP 值矩阵
} else {
  stop("shap_xgboost 不是有效的 shapviz 对象。请检查对象类型。")
}
shap_values <- as.matrix(shap_values)
shap_values <- apply(shap_values, 2, as.numeric)
# 计算 SHAP 值的绝对值之和
shap_importance <- colSums(abs(shap_values))

# 计算每个变量的影响力所占百分比
shap_percentage <- 100 * shap_importance / sum(shap_importance)

# 创建数据框，用于绘制柱状图
shap_df <- data.frame(
  Variable = names(shap_importance),
  Importance = shap_importance,
  Percentage = shap_percentage
)

# 假设 shap_df 已包含 'Variable' 和 'Importance' 列，现在添加一个 'Type' 列
shap_df$Type <- ifelse(shap_df$Variable %in% c("VHI", "VMI", "RS", "GVI", "SVI", "SDI"), 
                       "Microscale BE", "Macroscale BE")

# 绘制 SHAP 变量重要性柱状图，并显示百分比标签
ggplot(shap_df, aes(x = reorder(Variable, Importance), y = Importance, fill = Type)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Macroscale BE" = "#b71c2c", "Microscale BE" = "#e58166")) +
  coord_flip() +
  labs(title = "SHAP Variable Importance",
       x = "Variable",
       y = "Importance") +
  geom_text(aes(label = sprintf("%.2f%%", Percentage)), 
            hjust = -0.1, 
            size = 3.5) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


###
#单样本图
# 计算 SHAP 值
shap_values <- shapviz(fit_xgb_reg, X_pred = data_trainx)
# 生成综合 SHAP force plot
sv_force(shap_values, row_id = 100)

#非线性曲线图
# 计算 SHAP 值
shap_values <- shapviz(fit_xgb_reg, X_pred = as.matrix(data_trainx[,1:29]))
# 绘制 SHAP 依赖图
sv_dependence(shap_values, v = "DL")
# 绘制 SHAP 依赖图和拟合曲线
x <- data_trainx[, "DL"]
shap_values_v <- shap_values$S[, "DL"]
fit_loess <- loess(shap_values_v ~ x, span = 0.2)
predicted_shap <- predict(fit_loess, x)
# 绘制 
ggplot(data.frame(x = x, shap_values_v = shap_values_v, predicted_shap = predicted_shap), aes(x = x)) +
  geom_point(aes(y = shap_values_v), color = "#165290", alpha = 0.5) +  # 绘制原始 SHAP 值的散点图
  geom_line(aes(y = predicted_shap), color = "#b71c2c", size = 1) +  # 绘制拟合曲线
  labs(title = "",
       x = "Feature DL",
       y = "SHAP Value") +
  theme_minimal()
#让y轴更明显
ggplot(data.frame(x = x, shap_values_v = shap_values_v, predicted_shap = predicted_shap), aes(x = x)) +
  geom_point(aes(y = shap_values_v), color = "#165290", alpha = 0.5) +  # 绘制原始 SHAP 值的散点图
  geom_line(aes(y = predicted_shap), color = "#b71c2c", size = 1) +  # 绘制拟合曲线
  labs(title = "",
       x = "Feature DL",
       y = "SHAP Value") +
  ylim(-3, 3) +  # 设置y轴范围，手动调整范围以更好地显示曲线变化
  theme_minimal()
###
#局部shap值
# 加载完整原始数据集
original_data <- read.csv(file.choose())  # 更新文件路径

# 排除 "uid" 和 "WE" 列，获取预测数据
X_pred <- original_data %>% select(-uid, -WE)

# 确保数据是矩阵
data_X_pred <- as.matrix(X_pred)

data_DMatrix <- xgb.DMatrix(data = data_X_pred, label = original_data$WE)  # 将 label 指定为因变量 WE

#训练模型
fit_xgb_reg2<-xgb.train(data=data_DMatrix,eta=0.1 ,
                       gamma=0.001        ,
                       max_depth=8,
                       subsample=1,
                       colsample_bytree=0.5,               ,
                       objective="reg:squarederror", 
                       nrounds=1500,
                       watchlist = list(train = data_DMatrix),
                       verbose=1,
                       min_child_weight=3,
                       print_every_n = 100,
                       early_stopping_rounds = 400)



# 使用训练好的模型计算原始数据的 SHAP 值
# 假设fit_xgb_reg2是已训练的模型
shap_xgboost_full <- shapviz(fit_xgb_reg2, X_pred = data_X_pred)
str(shap_xgboost_full)

# 提取 SHAP 值矩阵
shap_values_matrix <- as.matrix(shap_xgboost_full$S)
nrow(shap_values_matrix)

# 计算每个样本的总 SHAP 值（考虑正负号）
total_shap_values <- rowSums(shap_values_matrix)

# 找到每个样本贡献最大的变量及其 SHAP 值
max_shap_indices <- apply(abs(shap_values_matrix), 1, which.max)
max_shap_variables <- colnames(shap_values_matrix)[max_shap_indices]
max_shap_values <- shap_values_matrix[cbind(1:nrow(shap_values_matrix), max_shap_indices)]

# 创建输出数据框，将 SHAP 值与原始数据合并
output_df <- data.frame(
  UID = original_data$uid,  # 替换为实际的唯一标识符列名
  Total_SHAP = total_shap_values,
  Max_Impact_Variable = max_shap_variables,
  Max_Impact_Value = max_shap_values
)
print(nrow(output_df))  # 应该是 3557
# 查看输出
head(output_df)
# 将结果写入 CSV 文件
write.csv(output_df, "shap_summary_output7.csv", row.names = FALSE)

# 查看输出
head(output_df)
###

####

# 创建 SHAP 值对象，启用交互效果
shap_values <- shapviz(fit_xgb_reg, X_pred = as.matrix(data_trainx[,1:27]), interactions = TRUE)

# 提取 SHAP 交互效应矩阵
interaction_effects <- shap_values$S_inter

# 将 interaction_effects 转换为数据框，并确保 Interaction_Shap_Value 是数值
interaction_df <- as.data.frame(as.table(as.matrix(interaction_effects)))
colnames(interaction_df) <- c("Feature_1", "Feature_2", "Interaction_Shap_Value")

# 确保 Interaction_Shap_Value 列为数值
interaction_df$Interaction_Shap_Value <- as.numeric(interaction_df$Interaction_Shap_Value)

# 绘制蜂巢图
ggplot(interaction_df, aes(x = Feature_1, y = Feature_2, fill = Interaction_Shap_Value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "SHAP Interaction Values", x = "Feature 1", y = "Feature 2", fill = "Interaction\nSHAP Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
# Display summary plot for SHAP interaction values (替代性实现)
sv_interaction(shap_values) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

##

####
# 计算 SHAP 主效应
shap_values <- shapviz(fit_xgb_reg, X_pred = as.matrix(data_trainx))
shap_main_effect <- colMeans(abs(shap_values$S))  # 计算每个特征 SHAP 值的均值作为主效应重要性
# 假设 shap_values$S 是一个包含主效应 SHAP 值的矩阵 (nrow(data_trainx) x n_features)
n_features <- ncol(data_trainx)
shap_interaction_values <- array(0, dim = c(nrow(data_trainx), n_features, n_features))
print(shap_main_effect)
# 计算交互效应
for (i in 1:(n_features - 1)) {
  for (j in (i + 1):n_features) {
    # 对于每个样本，计算特征i和特征j的交互效应
    shap_interaction_values[, i, j] <- shap_values$S[, i] * shap_values$S[, j]
  }
}

# 计算交互效应的均值
shap_interaction_effect <- apply(shap_interaction_values, c(2, 3), mean)

# 主效应数据框
main_effect_df <- data.frame(
  Feature = colnames(data_trainx),
  Importance = shap_main_effect
)

# 交互效应数据框，保留上三角部分（排除重复项）
interaction_effect_df <- data.frame(
  Feature = outer(colnames(data_trainx), colnames(data_trainx), paste, sep = " x ")[upper.tri(shap_interaction_effect)],
  Importance = shap_interaction_effect[upper.tri(shap_interaction_effect)]
)
# 选择前 12 个最重要的交互效应
top_main <- main_effect_df[order(main_effect_df$Importance, decreasing = TRUE), ][1:12, ]
top_interactions <- interaction_effect_df[order(interaction_effect_df$Importance, decreasing = TRUE), ][1:18, ]
print(top_interactions)
# 绘制 SHAP 主效应柱状图
p1 <- ggplot(top_main, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#b71c2c") +
  coord_flip() +
  labs(title = "Top 12 SHAP Main Effect Importance",
       x = "Feature",
       y = "SHAP Importance") +
  theme_minimal()

# 绘制 SHAP 交互效应柱状图，显示前 12 个
p2 <- ggplot(top_interactions, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#e58166") +
  coord_flip() +
  labs(title = "Top 12 SHAP Interaction Effect Importance",
       x = "Feature Pair",
       y = "SHAP Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # 调整标签角度以避免重叠

# 分开显示两个图
print(p1)
print(p2)

###
library(shapviz)
library(ggplot2)

# 1. 计算 SHAP 主效应
shap_values <- shapviz(fit_xgb_reg, X_pred = as.matrix(data_trainx))
shap_main_effect <- colMeans(abs(shap_values$S))  # 计算主效应均值

# 检查交互效应矩阵是否存在
if (is.null(shap_values$S_inter)) {
  stop("The shapviz object does not contain interaction effects. Check the model or consider using another tool like Python SHAP.")
}

# 2. 提取交互效应矩阵
interaction_matrix <- shap_values$S_inter  # 假设 S_inter 包含交互效应

# 替换交互效应矩阵中的 NA
interaction_matrix[is.na(interaction_matrix)] <- 0

# 检查特征名称
if (any(is.na(colnames(data_trainx)))) {
  stop("Feature names contain NA. Please check your dataset column names.")
}

# 3. 计算交互效应的重要性（绝对值的均值）
shap_interaction_effect <- apply(abs(interaction_matrix), c(2, 3), mean)

# 4. 主效应数据框
main_effect_df <- data.frame(
  Feature = colnames(data_trainx),
  Importance = shap_main_effect
)

# 5. 交互效应数据框，保留上三角部分（排除重复项）
interaction_effect_df <- data.frame(
  Feature = outer(colnames(data_trainx), colnames(data_trainx), paste, sep = " x ")[upper.tri(shap_interaction_effect)],
  Importance = shap_interaction_effect[upper.tri(shap_interaction_effect)]
)

# 6. 选择前 12 个最重要的主效应和交互效应
top_main <- main_effect_df[order(main_effect_df$Importance, decreasing = TRUE), ][1:15, ]
top_interactions <- interaction_effect_df[order(interaction_effect_df$Importance, decreasing = TRUE), ][1:15, ]

# 7. 绘制 SHAP 主效应柱状图
p1 <- ggplot(top_main, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#b71c2c") +
  coord_flip() +
  labs(title = "Top 12 SHAP Main Effect Importance",
       x = "Feature",
       y = "SHAP Importance") +
  theme_minimal()

# 8. 绘制 SHAP 交互效应柱状图
p2 <- ggplot(top_interactions, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#e58166") +
  coord_flip() +
  labs(title = "Top 12 SHAP Interaction Effect Importance",
       x = "Feature Pair",
       y = "SHAP Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # 调整标签角度以避免重叠

# 分开显示两个图
print(p1)
print(p2)

####
#shap两变量交互作用热力图
#散点图
plot_shap_interaction <- function(data, shap_values, feature1, feature2) {
  # 获取特征索引
  index_1 <- which(colnames(data) == feature1)
  index_2 <- which(colnames(data) == feature2)
  
  # 提取交互效应值
  interaction_values <- shap_values[, index_1, index_2]
  
  # 创建数据框，用于绘制图形
  interaction_data <- data.frame(
    Feature1 = data[, feature1],
    Feature2 = data[, feature2],
    SHAP_interaction = as.numeric(interaction_values)  # Ensure SHAP_interaction is numeric
  )
  
  # 检查数据类型和范围
  print(class(interaction_data$SHAP_interaction))  # Should print 'numeric'
  print(range(interaction_data$SHAP_interaction, na.rm = TRUE))  # Check if values are continuous
  
  # 绘制散点图，使用连续色标
  ggplot(interaction_data, aes(x = Feature1, y = Feature2, color = SHAP_interaction)) +
    geom_point(alpha = 0.7) +  # Use points for better visualization
    scale_color_gradientn(colors = c("#165290", "#4a90c3", "#fce3d4", "#efac87", "#c85440", "#b71c2c"),
                          name = "SHAP Interaction", limits = range(interaction_data$SHAP_interaction, na.rm = TRUE)) +  # Set continuous scale explicitly
    labs(title = paste("SHAP Interaction Plot between", feature1, "and", feature2),
         x = feature1,
         y = feature2) +
    theme_minimal() +
    theme(legend.position = "right")  # Adjust legend position
}

# 调用函数，只需替换变量名即可生成不同的交互效应图
plot_shap_interaction(data_trainx, shap_interaction_values, "DF", "SL")

####
#两变量热力方块图
plot_shap_interaction_heatmap <- function(data, shap_values, feature1, feature2, bins = 30) {
  # 获取特征索引
  index_1 <- which(colnames(data) == feature1)
  index_2 <- which(colnames(data) == feature2)
  
  # 提取交互效应值
  interaction_values <- shap_values[, index_1, index_2]
  
  # 创建数据框，用于绘制图形
  interaction_data <- data.frame(
    Feature1 = data[, feature1],
    Feature2 = data[, feature2],
    SHAP_interaction = as.numeric(interaction_values)  # Ensure SHAP_interaction is numeric
  )
  
  # 将数据进行分箱处理
  interaction_data <- interaction_data %>%
    mutate(
      Feature1_bin = cut(Feature1, breaks = bins),
      Feature2_bin = cut(Feature2, breaks = bins)
    ) %>%
    group_by(Feature1_bin, Feature2_bin) %>%
    summarize(Mean_SHAP_interaction = mean(SHAP_interaction, na.rm = TRUE), .groups = "drop")
  
  # 创建一个新数据框，包含分箱中心点，用于绘制热力图
  interaction_data <- interaction_data %>%
    mutate(
      Feature1_center = as.numeric(sub("\\((.+),.*", "\\1", Feature1_bin)) + 
        (as.numeric(sub(".*,(.+)\\]", "\\1", Feature1_bin)) - 
           as.numeric(sub("\\((.+),.*", "\\1", Feature1_bin))) / 2,
      Feature2_center = as.numeric(sub("\\((.+),.*", "\\1", Feature2_bin)) + 
        (as.numeric(sub(".*,(.+)\\]", "\\1", Feature2_bin)) - 
           as.numeric(sub("\\((.+),.*", "\\1", Feature2_bin))) / 2
    )
  
  # 绘制热力图
  ggplot(interaction_data, aes(x = Feature1_center, y = Feature2_center, fill = Mean_SHAP_interaction)) +
    geom_tile() +
    scale_fill_gradientn(colors = c("#165290", "#95C4E0","#fce3d4", "#efac87", "#c85440", "#b71c2c"),
                         name = "SHAP Interaction",
                         limits = range(interaction_data$Mean_SHAP_interaction, na.rm = TRUE)) +  # Set continuous scale explicitly
    labs(title = paste("SHAP Interaction Heatmap between", feature1, "and", feature2),
         x = feature1,
         y = feature2) +
    theme_minimal() +
    theme(legend.position = "right")  # Adjust legend position
}

# 调用函数
plot_shap_interaction_heatmap(data_trainx, shap_interaction_values, "BD", "SL", bins = 30)
####

####
#热力方块，global—limits可调范围
plot_shap_interaction_heatmap <- function(data, shap_values, feature1, feature2, bins = 30, global_limits = c(-5, 5)) {
  # 获取特征索引
  index_1 <- which(colnames(data) == feature1)
  index_2 <- which(colnames(data) == feature2)
  
  # 提取交互效应值
  interaction_values <- shap_values[, index_1, index_2]
  
  # 创建数据框，用于绘制图形
  interaction_data <- data.frame(
    Feature1 = data[, feature1],
    Feature2 = data[, feature2],
    SHAP_interaction = as.numeric(interaction_values)  # Ensure SHAP_interaction is numeric
  )
  
  # 将数据进行分箱处理
  interaction_data <- interaction_data %>%
    mutate(
      Feature1_bin = cut(Feature1, breaks = bins),
      Feature2_bin = cut(Feature2, breaks = bins)
    ) %>%
    group_by(Feature1_bin, Feature2_bin) %>%
    summarize(Mean_SHAP_interaction = mean(SHAP_interaction, na.rm = TRUE), .groups = "drop")
  
  # 创建一个新数据框，包含分箱中心点，用于绘制热力图
  interaction_data <- interaction_data %>%
    mutate(
      Feature1_center = as.numeric(sub("\\((.+),.*", "\\1", Feature1_bin)) + 
        (as.numeric(sub(".*,(.+)\\]", "\\1", Feature1_bin)) - 
           as.numeric(sub("\\((.+),.*", "\\1", Feature1_bin))) / 2,
      Feature2_center = as.numeric(sub("\\((.+),.*", "\\1", Feature2_bin)) + 
        (as.numeric(sub(".*,(.+)\\]", "\\1", Feature2_bin)) - 
           as.numeric(sub("\\((.+),.*", "\\1", Feature2_bin))) / 2
    )
  
  # 绘制热力图
  ggplot(interaction_data, aes(x = Feature1_center, y = Feature2_center, fill = Mean_SHAP_interaction)) +
    geom_tile() +
    scale_fill_gradientn(colors = c("#165290", "#4a90c3", "#fce3d4", "#efac87", "#c85440", "#b71c2c"),
                         name = "SHAP Interaction",
                         limits = global_limits) +  # 设置固定的颜色范围
    labs(title = paste("SHAP Interaction Heatmap between", feature1, "and", feature2),
         x = feature1,
         y = feature2) +
    theme_minimal() +
    theme(legend.position = "right")  # 调整图例位置
}

# 调用函数
plot_shap_interaction_heatmap(data_trainx, shap_interaction_values, "BD", "SL", bins = 30)

####
#三维散点图
library(plotly)

plot_shap_interaction_3d <- function(data, shap_values, feature1, feature2) {
  index_1 <- which(colnames(data) == feature1)
  index_2 <- which(colnames(data) == feature2)
  
  interaction_values <- shap_values[, index_1, index_2]
  
  interaction_data <- data.frame(
    Feature1 = data[, feature1],
    Feature2 = data[, feature2],
    SHAP_interaction = as.numeric(interaction_values)
  )
  
  # 绘制 3D 散点图
  plot_ly(
    interaction_data, 
    x = ~Feature1, 
    y = ~Feature2, 
    z = ~SHAP_interaction, 
    type = "scatter3d", 
    mode = "markers",
    marker = list(
      color = ~SHAP_interaction, 
      colorscale = c("#165290", "#b71c2c"), 
      colorbar = list(title = "SHAP Interaction")
    )
  ) %>%
    layout(
      title = paste("3D SHAP Interaction between", feature1, "and", feature2),
      scene = list(
        xaxis = list(title = feature1),
        yaxis = list(title = feature2),
        zaxis = list(title = "SHAP Interaction")
      )
    )
}

# 调用函数
plot_shap_interaction_3d(data_trainx, shap_interaction_values, "DW", "NDVI")

###
# 加载必要的库
# 加载必要的包
library(MBA)
library(plotly)

plot_shap_interaction_topview <- function(data, shap_values, feature1, feature2) {
  # 获取特征列的索引
  index_1 <- which(colnames(data) == feature1)
  index_2 <- which(colnames(data) == feature2)
  
  # 提取 SHAP 交互作用值
  interaction_values <- shap_values[, index_1, index_2]
  
  # 检查并处理负值和零值
  interaction_values <- ifelse(interaction_values <= 0, 1e-5, interaction_values)
  
  # 对 SHAP 值进行平滑处理，避免极端值的影响
  interaction_values <- log1p(interaction_values)  # 取对数平滑，防止极值过大
  
  # 将数据整理为数据框
  interaction_data <- data.frame(
    Feature1 = data[, feature1],
    Feature2 = data[, feature2],
    SHAP_interaction = as.numeric(interaction_values)
  )
  
  # 使用 MBA 插值平滑数据
  mba_result <- mba.surf(
    interaction_data[, c("Feature1", "Feature2", "SHAP_interaction")], 
    no.X = 500,  # 增加网格数量以提高平滑度
    no.Y = 500   # 增加网格数量以提高平滑度
  )
  
  # 提取插值后的数据
  interp_data <- mba_result$xyz.est
  x <- interp_data$x
  y <- interp_data$y
  z <- t(interp_data$z)  # 需要转置 Z 值，以适应 plotly 的格式
  
  # 绘制二维顶视图
  fig <- plot_ly(
    x = x, 
    y = y, 
    z = z, 
    type = "contour",  # 使用等高线图表示二维平面数据
    colorscale = c("#4a90c3", "#fce3d4", "#efac87", "#c85440", "#b71c2c"),
    colorbar = list(title = "SHAP Interaction")
  ) %>%
    layout(
      title = paste("2D SHAP Interaction between", feature1, "and", feature2),
      xaxis = list(title = feature1),
      yaxis = list(title = feature2),
      showlegend = FALSE  # 移除图例
    )
  
  # 显示图形
  fig
}

# 示例调用（假设你已有 data_trainx 和 shap_interaction_values 数据）
plot_shap_interaction_topview(data_trainx, shap_interaction_values, "DF", "SL")

####
#交互矩阵热力图
# 假设 shap_main_effect 和 shap_interaction_effect 已经计算完成

# 获取特征数量和名称
n_features <- length(shap_main_effect)
feature_names <- colnames(data_trainx)  # 假设数据集的列名是特征名称

# 创建空矩阵
combined_matrix <- matrix(0, ncol = n_features, nrow = n_features, 
                          dimnames = list(feature_names, feature_names))

# 填充对角线为主效应值
diag(combined_matrix) <- shap_main_effect

# 只填充上三角部分的交互效应值
combined_matrix[upper.tri(combined_matrix)] <- shap_interaction_effect[upper.tri(shap_interaction_effect)]

# 转换为数据框以便于绘制
combined_df <- as.data.frame(as.table(combined_matrix))
colnames(combined_df) <- c("Feature1", "Feature2", "SHAP_Value")

# 去除下三角部分（因为值是0），保留对角线和下三角部分的值
combined_df <- combined_df[combined_df$SHAP_Value != 0, ]

# 绘制热力图，仅显示上三角和对角线
p_combined <- ggplot(combined_df, aes(x = Feature1, y = Feature2, fill = SHAP_Value)) +
  geom_tile() +
  geom_text(aes(label = round(SHAP_Value, 2)), color = "white", size = 3) +  # 显示 SHAP 值
  scale_fill_gradientn(colors = c("#165290", "#95C4E0", "#fce3d4", "#efac87", "#c85440", "#b71c2c")) +  # 从浅蓝色到黄色再到深桃色的渐变
  labs(title = "SHAP Main and Interaction Effects Heatmap",
       x = "Feature 1",
       y = "Feature 2",
       fill = "SHAP Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 打印热力图
print(p_combined)

####局部交互效应表
# 安装和加载必要的包
# 假设 shap_values_matrix 和 shap_interaction_values 已经计算完成
# shap_values_matrix 是主效应的 SHAP 值矩阵
# shap_interaction_values 是交互效应矩阵 (样本数 × 特征数 × 特征数)
# 加载完整原始数据集
# 加载完整原始数据集
o# 加载完整原始数据集
original_data <- read.csv(file.choose())  # 替换为实际路径

# 排除 "uid" 和 "WE" 列，获取预测数据
X_pred <- original_data %>% select(-uid, -WE)

# 确保数据是矩阵
data_X_pred <- as.matrix(X_pred)

# 构建 DMatrix，包含所有数据和因变量
data_DMatrix <- xgb.DMatrix(data = data_X_pred, label = original_data$WE)  # WE 为因变量

# 训练模型，基于完整数据集
fit_xgb_reg <- xgb.train(
  data = data_DMatrix,
  eta = 0.1,
  gamma = 0.001,
  max_depth = 8,
  subsample = 1,
  colsample_bytree = 0.5,
  objective = "reg:squarederror",
  nrounds = 1500,
  watchlist = list(train = data_DMatrix),
  verbose = 1,
  min_child_weight = 3,
  print_every_n = 100,
  early_stopping_rounds = 400
)

# 计算 SHAP 值，基于完整数据集
shap_xgboost_full <- shapviz(fit_xgb_reg, X_pred = data_X_pred)

# 提取 SHAP 值矩阵
shap_values_matrix <- as.matrix(shap_xgboost_full$S)

# 获取特征数量
n_features <- ncol(shap_values_matrix)

# 初始化交互效应矩阵 (样本数 x 特征数 x 特征数)
shap_interaction_values <- array(0, dim = c(nrow(shap_values_matrix), n_features, n_features))

# 计算交互效应矩阵
for (i in 1:(n_features - 1)) {
  for (j in (i + 1):n_features) {
    # 交互效应计算：特征 i 和 j 的 SHAP 值乘积
    shap_interaction_values[, i, j] <- shap_values_matrix[, i] * shap_values_matrix[, j]
  }
}

# 计算每个样本的交互效应总值（保留正负性）
interaction_sums <- apply(shap_interaction_values, 1, function(grid_interactions) {
  sum(grid_interactions[upper.tri(grid_interactions)])  # 上三角的值直接求和（保留正负）
})

# 找到每个样本的主导交互效应及其值
max_interaction_indices <- apply(shap_interaction_values, 1, function(grid_interactions) {
  max_idx <- which.max(abs(grid_interactions[upper.tri(grid_interactions)]))  # 绝对值最大索引
  list(
    idx = max_idx,
    value = grid_interactions[upper.tri(grid_interactions)][max_idx]  # 对应的实际值
  )
})

# 提取变量名组合（交互变量对）
interaction_names <- combn(colnames(shap_values_matrix), 2, paste, collapse = " x ")

# 获取主导交互变量对和交互值
dominant_interactions <- sapply(max_interaction_indices, function(item) {
  interaction_names[item$idx]
})
dominant_values <- sapply(max_interaction_indices, function(item) {
  item$value
})

# 确保 UID 长度匹配
uid_adjusted <- original_data$uid[1:length(interaction_sums)]

# 创建输出数据框
interaction_output_df <- data.frame(
  UID = uid_adjusted,  # 替换为实际的 UID 列名
  Total_Interaction = interaction_sums,
  Dominant_Interaction = dominant_interactions,
  Dominant_Value = dominant_values
)

# 查看输出结果
print(nrow(interaction_output_df))  # 应该与原始数据样本数一致
head(interaction_output_df)

# 将结果写入 CSV 文件
write.csv(interaction_output_df, "shap_interaction_output.csv", row.names = FALSE)

# 检查 SLxDF 在所有网格中的交互效应值
SLxDF_values <- shap_interaction_values[, "SL", "DF"]
summary(SLxDF_values)  # 查看 SLxDF 的统计信息

# 查找 SLxDF 在网格中是否有显著值
dominant_SLxDF_grids <- which(abs(SLxDF_values) == apply(abs(shap_interaction_values), 1, max))
print(dominant_SLxDF_grids)  # 输出主导 SLxDF 的网格


