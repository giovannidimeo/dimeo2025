library(haven)
library(did)
library(tidyverse)
library(xtable)
library(foreign)
library(extrafont)
library(Cairo)
library(cowplot)
library(purrr)
library(here)
library(plm)
library(lfe)
library(readxl)
library(future)
library(furrr)
library(tictoc)
library(data.table)
library(purrr)
options(datatable.rbindlist.check = "none") # To silence the rbindlist() related warning. The warning has no effect on the estimation, and occurs only because "did" uses data.table in the latest version.

# Set directory 
results <- ("~\results")

#_________________________________________________________________________________#
# LOAD DATA
#---------------------------------------------------------------------------------#

setwd(results)
data <- read.csv("dataready.csv")

data <- data %>%
  mutate(group = ifelse(counterfactual == 0, id, id - 10000))  %>%
  filter(owner == 0 | (owner == 1 & purchase > 1))

data_reduced <- data %>% # baseline results
  select(group, time, tot_consumption, YT, ST, tot_wealth, purchase, counterfactual, owner) %>%
  reshape(idvar = c("group", "time"),
                        timevar = "counterfactual", direction = "wide", 
                        sep = "") %>% 
  mutate(diffst = ST0 - ST1)  %>% 
  mutate(diffwt = tot_wealth0 - tot_wealth1) %>%
  mutate(diffct = tot_consumption0 - tot_consumption1)

data_at <- data %>% # conditional participation to stock market
  select(group, time, YT, AT, AP, ST, tot_wealth, purchase, counterfactual, owner) %>%
  mutate(stat = ST * AT) %>%
  mutate(stat_rel = stat/tot_wealth) %>%
  reshape(idvar = c("group", "time"),
                        timevar = "counterfactual", direction = "wide", 
                        sep = "") %>% 
  filter(AP0 == 1)  %>% 
  mutate(diffat = AT0 - AT1) %>% 
  mutate(diffstat_rel = stat_rel0 - stat_rel1) 

data_at_pre <-  data %>% # conditional participation to stock market before house purchase
  mutate(stat = ST * AT) %>%
  mutate(event = ifelse(owner == 1, time - purchase, NA)) %>%
  mutate(flag = ifelse(event == - 1 & AP == 1, 1, 0)) %>%
  mutate(flag = ifelse(owner == 0 & AP == 1, 99, flag)) %>%
  group_by(id) %>%
  mutate(flag = max(flag)) %>%
  mutate(stat1 = dplyr::lag(stat, n= 1, default = 0)) %>%
  ungroup() %>%
  filter(flag != 0) %>%
  mutate(returns = stat1*RT) %>%
  mutate(returns_cum = ave(returns, id, FUN = cumsum)) %>%
  select(group, time, YT, AT, ST, tot_wealth, purchase, stat, returns_cum, counterfactual, owner) %>%
  pivot_wider(id_cols = c("group", "time"),
              names_from = "counterfactual", 
              values_from = c(YT, AT, ST, tot_wealth, stat, returns_cum, purchase, owner),
              names_sep = "") %>%
  mutate(diffstat = stat0 - stat1)  %>%
  mutate(diffcumsum = returns_cum0 - returns_cum1)

#_________________________________________________________________________________#
# WRITE CUSTOM CSDID FUNCTION
#---------------------------------------------------------------------------------#
span_top <- 30
span_bottom <- 5
levels <- c(0.01, 0.05, 0.1)

cs_fun <- function(y, df_data, alpha) {
  
  set.seed(1215) # Setting seed in the function. This should prevent us generating results w/o explicitly setting the right seed. 
  
  # Estimate the ATT(g,t)
  result <- att_gt(yname = y,
                   gname = "purchase0",
                   idname = "group",
                   tname = "time",
                   est_method = "ipw",
                   control_group = "nevertreated",
                   panel = TRUE,
                   allow_unbalanced_panel = TRUE,
                   pl = FALSE,
                   xformla = ~ YT0, 
                   cores = 12,
                   alp = alpha,
                   base_period = "universal",
                   data = df_data)
  
  # Aggregate to ES coefficients
  es <- aggte(result, type = "dynamic", na.rm = TRUE, min_e = -span_bottom, max_e = span_top)
  
  return(list(result = result, es = es))
}

# _________________________________________________________________________________#
# RUN REGRESSIONS
#---------------------------------------------------------------------------------#
# Set # of cores to be used for parallelization
plan(multisession, workers = future::availableCores() - 5) # leave one for other system tasks

tic()

# REDUCED
varnames_bl <- c(
  "diffst", 
  "diffwt",
  "diffct"
)

# DATA_AT
varnames_at <- c("diffat",
              "diffstat_rel")


# DATA_AT_PRE
varnames_at_pre <- c("diffstat",
              "diffcumsum")


# Specify the datasets and variables for main and partner analyses
analysis_settings <- list(
  baseline = list(
    data = data_reduced,  # DEFINE DATA SET
    varnames = varnames_bl  # DEFINE VARIABLE LIST
  ), 
  cond_part = list(
    data = data_at,
    varnames = varnames_at
  ),
  cond_prior_part = list(
    data = data_at_pre, 
    varnames = varnames_at_pre
  )
)


# Initialize an empty list to store results for all variables and significance levels
MainResults_Megaloop <- list()

# Use future_map to iterate over analysis types in parallel
results_list <- future_map(names(analysis_settings), ~{
  type <- .x
  current_setting <- analysis_settings[[type]]
  current_data <- current_setting$data
  current_varnames <- current_setting$varnames
  
  # Initialize a list to store results for the current analysis type
  analysis_results <- list()
  
  # Iterate over variables (this could also be parallelized with future_map if desired)
  for (var in current_varnames) {
    es_01 <- NULL
    es_05 <- NULL
    es_10 <- NULL
    
    # Run the analysis for each significance level
    for (l in levels) {
      result <- cs_fun(var, current_data, l)
      es <- result$es
      
      # Store results based on significance level
      if (l == 0.01) {
        es_01 <- es
      } else if (l == 0.05) {
        es_05 <- es
      } else if (l == 0.1) {
        es_10 <- es
      }
    }
    
    # Create the table for a given variable
    table_name <- paste0(type, "_", var)
    table_data <- data.frame(
      group = es_05$egt,
      effect = es_05$att.egt,
      se = es_05$se.egt,
      lower = es_05$att.egt - es_05$crit.val.egt * es_05$se.egt,
      upper = es_05$att.egt + es_05$crit.val.egt * es_05$se.egt,
      critval99 = es_01$crit.val.egt,
      critval95 = es_05$crit.val.egt,
      critval90 = es_10$crit.val.egt,
      att = es_05$overall.att,
      attse = es_05$overall.se,
      unique_pids = es_05$DIDparams$n
    )
    # Store the table in the analysis_results list
    analysis_results[[table_name]] <- table_data
  }
  analysis_results
}, .progress = TRUE)  # Optional: show progress

# Combine results from all analyses into a single list
MainResults_Megaloop <- do.call(c, results_list)

toc()

# Select only those beginning with 'tot_' or 'partner_'
objects_to_save <- ls(pattern = "^(MainResults_Megaloop)")
setwd(results)
save(list = objects_to_save, file = paste0("AllResults", ".RData"))

# _________________________________________________________________________________#
# CREATE TABLES
#---------------------------------------------------------------------------------#

myround <- function(x) {
  ifelse(x < 1 & x > -1, round(x, 4), round(x, 2) )
}

i <- 1L

stars <- function(df) {
  df$tstat[i] <- 0
  df$tstatatt[i] <- 0
  for (i in 1:36) {
    print(i)
    df$tstat[i] <-   abs(df$effect[i]/df$se[i])
    if (!is.na(df$tstat[i]) &  df$tstat[i] >= df$critval99[i]) {
      df$stars[i] <- "***"
    } else if ( !is.na(df$tstat[i])  &  df$tstat[i] < df$critval99[i] & df$tstat[i] >= df$critval95[i]) {
      df$stars[i] <- "**"
    } else if (!is.na(df$tstat[i])  &  df$tstat[i] < df$critval95[i] & df$tstat[i] >= df$critval90[i]) {
      df$stars[i] <- "*"
    } else { 
      df$stars[i] <- "" }
    
    df$tstatatt[i] <-abs(df$att[i]/df$attse[i])
    
    if (!is.na(df$tstatatt[i]) &  df$tstatatt[i] >= qnorm(1-0.01/2)) {
      df$starsatt[i] <- "***"
    } else if ( !is.na(df$tstatatt[i])  &  df$tstatatt[i] < qnorm(1-0.01/2) & df$tstatatt[i] >= qnorm(1-0.05/2)) {
      df$starsatt[i] <- "**"
    } else if (!is.na(df$tstatatt[i])  &  df$tstatatt[i] < qnorm(1-0.05/2) & df$tstatatt[i] >= qnorm(1-0.1/2)) {
      df$starsatt[i] <- "*"
    } else { 
      df$starsatt[i] <- "" }
    
  }
  
  for (i in 1:36) {
    df$effect[i] <-  paste0(toString(myround(as.numeric(df$effect[i]))),  df$stars[i], sep = "")
    df$att[i] <-  paste0(toString(myround(as.numeric(df$att[i]))),  df$starsatt[i], sep = "")
    
  }
  return(df)
}

tab_function <- function(df, name){  
  df <- stars(df)
  
  # Initialize the first row (coefficients)
  table_tex1 <- data.frame("Dependent variable" = as.character(name),
                           check.names = FALSE)
  
  # Initialize the second row (standard errors)
  table_tex2 <- data.frame("Dependent variable" = "",
                           check.names = FALSE)
  
  # Add columns 0 to 30
  for (i in 0:30) {
    table_tex1[[as.character(i)]] <- df$effect[6 + i]
    table_tex2[[as.character(i)]] <- paste0("(", myround(df$se[6 + i]), ")")
  }
  
  # Add final columns
  table_tex1$Overall <- df$att[1]
  table_tex1$`Unique obs.` <- df$unique_pids[1]
  
  table_tex2$Overall <- paste0("(", myround(df$attse[1]), ")")
  table_tex2$`Unique obs.` <- ""
  
  tab <- rbind(table_tex1, table_tex2)
  return(tab)
}

generate_table <- function(list) {
  
  tab_temp <- data.frame(matrix(nrow = 65, ncol = length(list))) 
  
  for (v in 1:length(list)) {
    name <- paste0("(", as.character(v), ")")
    tab_or <- tab_function(list[[v]], name)
    for (m in 1:32) {
      k = m*2-1
      j = m+1
      tab_temp[[v]][k] <- tab_or[[j]][1]
      tab_temp[[v]][k+1] <- tab_or[[j]][2]
    }
    tab_temp[[v]][65] <-  tab_or[["Unique obs."]][1]
    colnames(tab_temp)[v] = name
  }
  
  tab_temp["Event time"] <- ""
  tab_temp  <- tab_temp %>% relocate("Event time")
  
  for (p in 1:32) {
    q = p*2 - 1
    tab_temp[["Event time"]][q] <- as.character(p-1)
    tab_temp[["Event time"]][q+1] = " "
  }
  tab_temp[["Event time"]][63] = "Overall"
  tab_temp[["Event time"]][65] = "Unique obs."
  
  print(xtable(tab_temp, type = "latex"), include.rownames=FALSE)
  print(tab_temp)
}


#Generate the table
var_list <- list( "(1)" = MainResults_Megaloop$baseline_diffst,
                  "(2)" = MainResults_Megaloop$baseline_diffwt,
                  "(3)" = MainResults_Megaloop$baseline_diffct,
                  "(4)" = MainResults_Megaloop$cond_part_diffat,
                  "(5)" = MainResults_Megaloop$cond_part_diffstat_rel,
                  "(6)" = MainResults_Megaloop$cond_prior_part_diffstat,
                  "(7)" = MainResults_Megaloop$cond_prior_part_diffcumsum)


generate_table(var_list)








