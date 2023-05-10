# Medindo a força de uma moeda (Real) em relação a outras moedas, para o preço de diferentes commodities

# Pacotes
install.packages("matrixStats")
install.packages("dplyr")
install.packages("timeSeries")
install.packages("StatMatch")
install.packages("FNN")
install.packages("diffusionMap")
install.packages("rARPACK")
install.packages("leaps")
install.packages("energy")
install.packages("pracma")
install.packages("rowr")
library(matrixStats)
library(dplyr)
library(timeSeries)
library(StatMatch)
library(FNN)
library(diffusionMap)
library(rARPACK)
library(leaps)
library(energy)
library(pracma)
library(rowr)


# Carregando os datasets
# Datasets com preços dos commodities e cotações de diferentes moedas para o mesmo período
currencies <- read.csv(file = "currencies.csv", header = TRUE)
commodities <- read.csv(file = "commodities.csv", header = TRUE)
View(currencies)
View(commodities)


# Merge dos datasets pela coluna de Data
complete <- merge(currencies, commodities, by = 'Date', sort = FALSE)
complete <- complete[complete.cases(complete),]
currencies <- complete[,2:ncol(currencies)]
commodities <- complete[,(ncol(currencies)+3):ncol(complete)]


# Função para aplicar o PCA para redução de dimensionalidade
run_PCA = function(X_train, nrFactors){
  
  myPCA = prcomp(X_train, center = TRUE, scale. = TRUE, retx = TRUE);
  pcaRot = myPCA$rotation
  EIGS_PCA = myPCA$x
  
  q = myPCA$sdev
  
  FACTORS = EIGS_PCA[, 1:nrFactors];
  
  return(FACTORS);
}


# Função para gerar o Plot
plot_cumsum = function(CUM_SUM_MTX,MAIN,low,high,location){

  nrTickers = dim(CUM_SUM_MTX)[2];
  nrDays = dim(CUM_SUM_MTX)[1];
  myColors = c('black', 'red', 'blue', 'green', 'yellow', 'magenta')
  
  plot (c(0, nrDays),c(low,high),type='n', xlab ='days', ylab = 'Prices',main =  MAIN) 
  for ( i in 1 : nrTickers){
    # Plots de timeseries no mesmo gráfico
    lines( 1: nrDays, CUM_SUM_MTX[ ,i], col = myColors[i], lwd=2);  
  }
  
  legend( location, legend = colnames(CUM_SUM_MTX), lty = 1, lwd = 2, col = myColors)
}


# Ajustando os dados
nrDays <- nrow(complete)
table <- NULL
windowdata <- complete[,-1] 
complete_dropdates <- windowdata
estimated_values <- data.frame(matrix(nrow = (nrDays - 101), ncol = 0))


# Regressão Linear e Correlação (com o coeficiente de pearson)
# O loop For vai percorrer todo o dataset, calcular os coeficientes de correlação
# e criar a regressão linear dos elementos para então fazer as previsões
for(j in 1:38){
  
  estimate <- data.frame()

  for(d in 101:(nrDays-1)){
    
    response <- complete_dropdates[,j]
    predict <- complete_dropdates[,]
    response <- response[(d-100):(d-1)]
    predict <- predict[(d-99):d,]
    
    correlation <- cor(as.matrix(complete_dropdates), method = "pearson")
    diag(correlation) <- 0
    abs_correlation <- abs(correlation)
    
    y <- sort(abs_correlation[,j], decreasing = TRUE)[1:10]
    pos <- which(abs_correlation[,j] %in% y)
    model.elements <- predict[,pos]
    
    model.data <- data.frame(response,predict)
    model <- lm(response~as.matrix(model.elements))
    v <- model$coefficients
    est <- as.matrix(complete_dropdates[d,pos]) %*% v[2:11] + v[1]
    estimate <- rbind(estimate,est)
    
  }
  
estimated_values <- data.frame(estimated_values,estimate)
  
}


# Grava o resultado em uma matriz
estimated_values <- as.matrix(estimated_values)


# Regressão Linear pelo método Stepwise
estimated_values_stepwise <- data.frame(matrix(nrow = (nrDays - 101), ncol = 0))


# Repete o loop, mas agora fazendo as previsões com outro método
# Ao final faremos as previsões com diferentes métodos a fim de comparar a perfomance das moedas 
# em relação ao preço dos commodities
for(j in 1:38){
  
  estimate <- data.frame()
  
  for(d in 101:(nrDays-1)){
    
    response <- complete_dropdates[,j]
    predict <- complete_dropdates[,]
    response <- response[(d-100):(d-1)]
    predict <- predict[(d-99):(d),]
    
    # Calcula as correlações para o período
    model.data <- data.frame(response, predict)
    backwards <- regsubsets(response ~ ., data=model.data, method = "forward", nvmax = 40)
    topten <- summary(backwards)$which[10,]
    model.elements <- predict[,topten[2:39]]
    
    model<-lm(response~as.matrix(model.elements))
    v <- model$coefficients
    est <- as.matrix(complete_dropdates[d,topten[2:39]]) %*% v[2:11] + v[1]
    estimate <- rbind(estimate,est)
  }
  
  estimated_values_stepwise <- data.frame(estimated_values_stepwise, estimate)
  
}


# Aplicação dos modelos KNN, Regressão Linear e do PCA

# Definindo o tamanho da janela em dias
nrLags = 1
lookback = 100 
nrTickers <- ncol(windowdata)
knn_Pred <- NULL 
lm_Pred <- NULL


# Criando um array de zeros que vão receber as previsões do modelo KNN
knn_Pred = array(0, dim = c(nrDays,nrTickers))


# Criando um array de zeros que vão receber as previsões do modelo de Regressão Linear
lm_Pred = array(0, dim = c(nrDays,nrTickers)); 


# Loop para criação dos dados de treino e de teste e aplicação dos modelos
for(i in (lookback + nrLags) : (nrDays - nrLags)) {
  xtrain <- windowdata[ (i-lookback) : (i-1), ];
  xtest <- windowdata[ i, , drop=FALSE]
  
  both <- rbind(xtrain , xtest)
  
  # Executando o PCA (função criada acima) em uma matriz de 10 atributos para reduzir a dimensionalidade
  ans <-run_PCA(both, 10) 

  xtrain<-ans[1:lookback,]
  xtest<-ans[ (lookback + 1), ]
  
  for(j in 1:38){
    y_train = windowdata [ (i-lookback+1) : i,j]
    y_test = windowdata [i+1,j ]
    
    # Aplicando o modelo KNN
    y_hat = knn.reg(train = xtrain, test = xtest, y = y_train, k = 10);
    y_hat = y_hat$pred;
    
    knn_Pred[i,j] = y_hat
    
    # Aplicando o modelo de regressão linear
    model<-lm(y_train~xtrain)
    v <- model$coefficients
    est <- xtest %*% v[2:11] + v[1]
    
    lm_Pred[i,j] = est
    
  }
}


# Removendo os zeros da matriz
knn_Pred = knn_Pred[(lookback+1):(nrDays-1),]
lm_Pred = lm_Pred[(lookback+1):(nrDays-1),]


# Usando a função which para encontra índices especíicos de colunas com nomes das moedas
# Usaremos apenas um commoditie, o Petróleo (oil em inglês)
USD_pos <- which (colnames(windowdata)=="USD")
BRL_pos <- which (colnames(windowdata)=="BRL")
AUD_pos <- which (colnames(windowdata)=="AUD")
RUB_pos <- which (colnames(windowdata)=="RUB")
ARS_pos <- which (colnames(windowdata)=="ARS")
Oil_pos <- which (colnames(windowdata)=="Oil")


# Vetor de 6 números para usarmos os índices
posvector <-c(USD_pos, BRL_pos, AUD_pos, RUB_pos, ARS_pos, Oil_pos)


# Definindo a matriz que vai receber os erros das previsões
# As colunas terão como nome as moedas e o commodite sendo analisado
# As linhas terão como títulos os modelos preditivos criados
errormatrix <- matrix(nrow = 4, ncol = 6) 
rownames(errormatrix) <- c("knnPCA", "lmPCA", "corLM", "stepLM")
colnames(errormatrix) <- c("USD", "BRL", "AUD", "RUB", "ARS", "Oil")


# Definindo a matriz que vai receber a soma dos erros das previsões
# As colunas terão como nome as moedas e o commodite sendo analisado
# As linhas terão como títulos os modelos preditivos criados
sumerrormatrix <- matrix(nrow = 4, ncol = 6)
rownames(sumerrormatrix) <- c("knnPCA", "lmPCA", "corLM", "stepLM")
colnames(sumerrormatrix) <- c("USD","BRL","AUD","RUB","ARS","Oil")

# Matrix com as previsões
predictionmatrix <- matrix(nrow = 1084, ncol = 30)


# Definindo a área para os plots
par(mfrow=c(3,2))


# Inicializando o contador e coletando as previsões
# Serão carregadas 3 matrizes: uma com as previsões, uma com os erros e uma com a soma dos erros
# Ao final, serão gerados os plots
j <- 1
for(i in 1:length(posvector)){
  # Obtendo o índice do vetor na posição desejada
  currentPos<-posvector[i] 
  knnPCA <- knn_Pred[,currentPos]
  lmPCA <- lm_Pred[,currentPos]
  corLM <- estimated_values[,currentPos]
  stepLM <- estimated_values_stepwise[,currentPos]
  realCurrency <- windowdata[(102:1185),currentPos]
  
  predictionmatrix[,j] <- knnPCA; 
  predictionmatrix[,j+1] <- lmPCA; 
  predictionmatrix[,j+2] <- corLM; 
  predictionmatrix[,j+3] <- stepLM; 
  predictionmatrix[,j+4] <- realCurrency; 
  
  # Calcula os erros e gera uma matriz
  
  Error_knnPCA <- (realCurrency - knnPCA)^2
  Error_knnPCA <- sqrt(Error_knnPCA)
  Error_lmPCA <- (realCurrency - lmPCA)^2
  Error_lmPCA <- sqrt(Error_lmPCA)
  Error_corLM <- (realCurrency - corLM)^2
  Error_corLM <- sqrt(Error_corLM)
  Error_stepLM <- (realCurrency - stepLM)^2
  Error_stepLM <- sqrt(Error_stepLM)
  
  knnPCA_mean <- mean(Error_knnPCA); errormatrix[1,i] <- knnPCA_mean; sumerrormatrix[1,i] <- sum(Error_knnPCA)
  lmPCA_mean <- mean(Error_lmPCA); errormatrix[2,i] <- lmPCA_mean; sumerrormatrix[2,i] <- sum(Error_lmPCA)
  corLM_mean <- mean(Error_corLM); errormatrix[3,i] <- corLM_mean; sumerrormatrix[3,i] <- sum(Error_corLM)
  stepLM_mean <- mean(Error_stepLM); errormatrix[4,i] <- stepLM_mean; sumerrormatrix[4,i] <- sum(Error_stepLM)
  
  names <- names(errormatrix[1,])
  currentname <- names[i]
  mainname <- paste("Erro Médio",currentname)
  mainname2 <- paste("Erro Total",currentname)
  
  barplot(errormatrix[,i], names.arg=c("knnPCA","lmPCA","corLM","stepLM"), main = mainname )
  barplot(sumerrormatrix[,i], names.arg = c("knnPCA","lmPCA","corLM","stepLM"), main = mainname2 )
  
  j <- j + 5
  
}


# Obtém o erro médio e cria o plot
meanerror <- rowMeans(errormatrix)
barplot(meanerror, main = "Erro Médio Total")


# Definindo a área de desenho dos plots
par(mfrow=c(1,1))


# Criando cálculos das previsões para cada modelo criado
USDmatrix <- predictionmatrix[,1:5]
colnames(USDmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")
BRLmatrix <- predictionmatrix[,6:10]
colnames(BRLmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")
AUDmatrix <- predictionmatrix[,11:15]
colnames(AUDmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")
RUBmatrix <- predictionmatrix[,16:20]
colnames(RUBmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")
ARSmatrix <- predictionmatrix[,21:25]
colnames(ARSmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")
Oilmatrix <- predictionmatrix[,26:30]
colnames(Oilmatrix) <- c("knnPCA","lmPCA","corLM","stepLM","Preço Real")


# Plot
# Comparando a força do real em relação a outras moedas, para o preço de commodites
# Usamos diferentes modelos preditivos a fim de garantir precisão nas previsões
par(mfrow=c(1,1))
plot_cumsum(USDmatrix, "USD",90,120,"topright")
plot_cumsum(BRLmatrix,"BRL",68,110,"topleft")
plot_cumsum(AUDmatrix, "AUD",85,115,"topleft")
plot_cumsum(RUBmatrix,"RUB",70,140,"bottomright")
plot_cumsum(ARSmatrix,"ARS",65,120,"topleft")
plot_cumsum(Oilmatrix,"Oil",55,125,"bottomright")


