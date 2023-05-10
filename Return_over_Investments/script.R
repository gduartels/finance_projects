# Análise de Retorno de Investmentos com pacotes PortfolioAnalytics e PerformanceAnalytics

# Pacotes
install.packages("PortfolioAnalytics")
install.packages("quantmod")
install.packages("PerformanceAnalytics")
install.packages("zoo")
install.packages("plotly")
library(PortfolioAnalytics)
library(quantmod)
library(PerformanceAnalytics)
library(zoo)
library(plotly)


# Obtendo os dados de ações de empresas listadas na bolsa de valores americana
getSymbols(c("MSFT", "SBUX", "IBM", "AAPL", "^GSPC", "AMZN"))

# Criando um dataframe e ajustado o preço das ações
prices.data <- merge.zoo(MSFT[,6], SBUX[,6], IBM[,6], AAPL[,6], GSPC[,6], AMZN[,6])

# Calculando o retorno
returns.data <- CalculateReturns(prices.data)
returns.data <- na.omit(returns.data)

# Definindo os nomes
colnames(returns.data) <- c("MSFT", "SBUX", "IBM", "AAPL", "^GSPC", "AMZN")

# Salvando um vetor com a média de retorno e matriz de covariância
meanReturns <- colMeans(returns.data)
covMat <- cov(returns.data)

# Definindo o nome dos assets no portfólio de investimentos
# Isso define a especificação do portfólio
port <- portfolio.spec(assets = c("MSFT", "SBUX", "IBM", "AAPL", "^GSPC", "AMZN"))

# Adicionando constraints

# Box
port <- add.constraint(port, type = "box", min = 0.05, max = 0.8)

# Leverage
port <- add.constraint(portfolio = port, type = "full_investment")

# Gerando portfólios randômicos. Isso basicamente cria um conjunto de portfólio viáveis
# que satisfazem as condições (constraints)
# Lista de todas as constraints disponíveis:
# https://cran.r-project.org/web/packages/PortfolioAnalytics/vignettes/portfolio_vignette.pdf
rportfolios <- random_portfolios(port, permutations = 500000, rp_method = "sample")

# Obtendo a variância mínima do portfólio de investimentos
minvar.port <- add.objective(port, type = "risk", name = "var")
print(minvar.port)

# Otimizando
minvar.opt <- optimize.portfolio(returns.data, minvar.port, optimize_method = "random", 
                                 rp = rportfolios)
print(minvar.opt)

# Gerando o retorno máximo do portfólio de investimento
maxret.port <- add.objective(port, type = "return", name = "mean")

# Otimizando
maxret.opt <- optimize.portfolio(returns.data, maxret.port, optimize_method = "random", 
                                 rp = rportfolios)

# Gerando um vetor de retornos
minret <- 0.06/100
maxret <- maxret.opt$weights %*% meanReturns
vec <- seq(minret, maxret, length.out = 100)

# Agora que temos a variância mínima, bem como as carteiras de retorno máximo, 
# podemos construir a fronteira mais eficiente de investimento. 
# Vamos adicionar um objetivo de concentração de peso, bem como garantir que não teremos carteiras de investimento
# muito concentradas.
eff.frontier <- data.frame(Risk = rep(NA, length(vec)),
                           Return = rep(NA, length(vec)), 
                           SharpeRatio = rep(NA, length(vec)))

frontier.weights <- mat.or.vec(nr = length(vec), nc = ncol(returns.data))
colnames(frontier.weights) <- colnames(returns.data)

?optimize.portfolio

# Podemos utlizar o método de otimização random (mais eficiente) ou ROI (R Optmization Infrastructure)
# Usaremos o random, mas ele requer muito tempo de processamento (algumas horas)
for(i in 1:length(vec)){
  eff.port <- add.constraint(port, type = "return", name = "mean", return_target = vec[i])
  eff.port <- add.objective(eff.port, type = "risk", name = "var")

  eff.port <- optimize.portfolio(returns.data, eff.port, optimize_method = "random")
  
  eff.frontier$Risk[i] <- sqrt(t(eff.port$weights) %*% covMat %*% eff.port$weights)
  
  eff.frontier$Return[i] <- eff.port$weights %*% meanReturns
  
  eff.frontier$Sharperatio[i] <- eff.port$Return[i] / eff.port$Risk[i]
  
  frontier.weights[i,] = eff.port$weights
  
  print(paste(round(i/length(vec) * 100, 0), "% concluído..."))
}

feasible.sd <- apply(rportfolios, 1, function(x){
  return(sqrt(matrix(x, nrow = 1) %*% covMat %*% matrix(x, ncol = 1)))
})

feasible.means <- apply(rportfolios, 1, function(x){
  return(x %*% meanReturns)
})

feasible.sr <- feasible.means / feasible.sd

# Plot com Plotly
p <- plot_ly(x = feasible.sd, y = feasible.means, color = feasible.sr, 
             mode = "markers", type = "scattergl", showlegend = F,
             
             marker = list(size = 3, opacity = 0.5, 
                           colorbar = list(title = "Sharpe Ratio"))) %>% 
  
  add_trace(data = eff.frontier, x = Risk, y = Return, mode = "markers", 
            type = "scattergl", showlegend = F, 
            marker = list(color = "#F7C873", size = 5)) %>% 
  
  layout(title = "Gráfico Plotly do Portfolios",
         yaxis = list(title = "Média de Retorno", tickformat = ".2%"),
         xaxis = list(title = "Desvio Padrão", tickformat = ".2%"),
         plot_bgcolor = "#434343",
         paper_bgcolor = "#F8F8F8",
         annotations = list(
           list(x = 0.4, y = 0.75, 
                ax = -30, ay = -30, 
                text = "Fronteira Eficiente", 
                font = list(color = "#F6E7C1", size = 15),
                arrowcolor = "white")
         ))

# Reshape
frontier.weights.melt <- reshape2::melt(frontier.weights)

# plot
q <- plot_ly(frontier.weights.melt, x = Var1, y = value, group = Var2, type = "bar") %>%
  layout(title = "Pesos dos Portfolios Através da Fronteira", barmode = "stack",
         xaxis = list(title = "Index"),
         yaxis = list(title = "Pesos(%)", tickformat = ".0%"))

# Plot
p
q
