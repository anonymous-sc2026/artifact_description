# Load required packages
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(readr)
library(psych)  # for corr.test
library(GGally) # for ggpairs
library(stringr)

setwd("./input/serving_request/emulation_results_openai")

generate_figs <- function(file){
  # file <- "results_deadline_mistral_nvidia-a100-80gb-pcie_1.csv"
  # file <- "results_deadline_sigma=0.0_nvidia-h100-nvl_1.csv"
  # setwd("./input/serving_request/emulation_results_with_percent")
  # file <- "results_deadline_mistral_nvidia-h100-nvl_1.csv"
  # setwd("./input/serving_request/emulation_results_openai")
  # file<-"results_deadline_qwen_nvidia-a100-80gb-pcie_4.csv"
  
  # --- Model ---
  model <- case_when(

      str_detect(file, "mistral") ~ "Mistral-7B",
    str_detect(file, "qwen") ~ "Qwen3-30B",
    TRUE ~ NA_character_
  )
  
  # --- GPU type ---
  gpu <- case_when(
    str_detect(file, "a100") ~ "A100 80GB",
    str_detect(file, "h100") ~ "H100 94GB",
    TRUE ~ NA_character_
  )
  
  # --- Number of GPUs ---
  n_gpus <- str_extract(file, "(?<=_)\\d+(?=\\.csv)")
  n_gpus <- paste0(n_gpus, ifelse(n_gpus == "1", " GPU", " GPUs"))
  
  xp_info <- paste(model, "-", gpu, "-", n_gpus)
  
  # 1. Load the CSV file
  
  
  xp_suffix <- sub("^results_deadline|\\.csv$", "", file)
  df <- read.csv(file)
  
  print(model)
  if (model == "Qwen3-30B") {
    df <- df %>% filter(sla_factor > 2)
  }
  
  df <- df %>% 
    filter(rt_scaling %in% c(1e-5, 0.001, 0.1)) %>% 
    filter(nb_req == 100) %>%
    filter(skip_lines %in% c(
      0, 50000, 100000, 150000, 200000, 250000, 300000, 350000,
      400000, 450000, 500000, 550000, 600000, 650000, 700000,
      750000, 800000, 850000, 900000, 950000, 1000000
    )) %>%
    filter(trace=="BurstGPT")
  
  
  df <- df %>%
    mutate(
      deadline_type = recode(
        deadline_type,
        "det-deadlines" = "Det.",
        "rnd-deadlines" = "Rnd"
      )
    ) 
  
  df$method <- dplyr::recode(df$method,
                             "baseline" = "vLLM sched.",
                             "out-of-order-discard-most-urgent" = "Deadline-aware",
                             "out-of-order-edf" = "EDF"
  )
  # Force the plotting order
  df$method <- factor(df$method, levels = c(
    "vLLM sched.",
    "Deadline-aware",
    "EDF"
  ))
  
  
  
  ############## UNSG #####################
  
  df_sub <- df %>% 
    select(method, rt_scaling, sla_factor, success_ratio, percent_urgent, skip_lines, deadline_type) %>%
    filter(percent_urgent != 0) %>%
    pivot_wider(names_from = method, values_from = success_ratio) %>%
    mutate(unsg_vLLM = 100*(`Deadline-aware`- `vLLM sched.` ) / percent_urgent) %>%
    mutate(unsg_EDF = 100*(`Deadline-aware`- `EDF` ) / percent_urgent) 
  
  df_unsg_rt <- df_sub %>%
    group_by(rt_scaling, deadline_type, percent_urgent) %>%
    summarise(
      mean_unsg_vllm = mean(unsg_vLLM, na.rm = TRUE),
      sd_unsg_vllm   = sd(unsg_vLLM, na.rm = TRUE),
      se_unsg_vllm   = sd_unsg_vllm / sqrt(n()),
      mean_unsg_edf  = mean(unsg_EDF, na.rm = TRUE),
      sd_unsg_edf    = sd(unsg_EDF, na.rm = TRUE),
      se_unsg_edf    = sd_unsg_edf / sqrt(n()),
      n              = n(),
      .groups = "drop"
    )

  
  # Réorganiser les données pour ggplot (long format)
  df_long_rt <- df_unsg_rt %>%
    pivot_longer(
      cols = c(mean_unsg_vllm, mean_unsg_edf, se_unsg_vllm, se_unsg_edf),
      names_to = c(".value", "method"),
      names_pattern = "(mean|se)_(unsg_.*)"
    ) %>%
    mutate(
      method = recode(method,
                      "unsg_vllm" = "vLLM",
                      "unsg_edf"  = "EDF")
    )
  
  ggplot(df_long_rt, aes(
    x = percent_urgent,
    y = mean,
    color = factor(rt_scaling),
    group = interaction(rt_scaling, method),
    linetype = method
  )) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(
      ymin = mean - se,
      ymax = mean + se,
      fill = factor(rt_scaling)
    ), alpha = 0.15, color = NA) +
    geom_hline(yintercept = 0, scales = "free_y", linetype = "dashed", color = "grey40") +
    facet_grid(deadline_type ~ ., labeller = labeller(
      deadline_type = function(x) paste("Deadline type =", x)
    )) +
    scale_linetype_manual(values = c("vLLM" = "solid", "EDF" = "dotted")) +
    scale_x_continuous(breaks = seq(10, 100, by = 10), limits = c(10, 100)) +
    labs(
      title = "UNSG for Different RT Scaling vs. Urgent Requests",
      subtitle = xp_info,
      x = "Percentage of urgent requests",
      y = "Urgency-normalized success gain (%)",
      color = "RT scaling",
      fill  = "RT scaling",
      linetype = "Method"
    ) +
    theme(legend.position = "top")
  
  # ggplot(
  #   df_unsg_rt,
  #   aes(
  #     x = percent_urgent,
  #     color = factor(rt_scaling),
  #     group = rt_scaling
  #   )
  # ) +
  #   geom_line(aes(y = mean_unsg, linetype = "baseline"), linewidth = 1) +
  #   geom_ribbon(
  #     aes(
  #       ymin = mean_unsg - se,
  #       ymax = mean_unsg + se,
  #       fill = factor(rt_scaling)
  #     ),
  #     alpha = 0.15,
  #     color = NA
  #   ) +
  #   
  #   geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  #   
  #   facet_grid(
  #     deadline_type ~ .,
  #     labeller = labeller(
  #       rt_scaling = function(x) paste("RT Scaling =", x),
  #       sla_factor = function(x) paste("SLA factor =", x),
  #       deadline_type = function(x) paste("Deadline type =", x)
  #     )
  #   ) +
  #   
  #   labs(
  #     title = "UNSG for Different RT Scaling in Function of Urgent Request",
  #     subtitle = xp_info,
  #     x = "Percentage of urgent requests",
  #     y = "Urgency-normalized success gain (%)",
  #     color = "RT scaling",
  #     fill  = "RT scaling",
  #     linetype = "Method"
  #   ) +
  #   
  #   scale_x_continuous(
  #     breaks = seq(10, 100, by = 10),
  #     limits = c(10, 100)
  #   ) +
  #   
  #   theme(legend.position = "top")
  
  ggsave(paste0("pdf_results/unsg_rt", xp_suffix, ".pdf"),width=8, height=4)
  
  df_unsg_sla <- df_sub%>%
    group_by(
      sla_factor,
      deadline_type,
      percent_urgent
    ) %>%
    summarise(
      mean_unsg_vllm = mean(unsg_vLLM, na.rm = TRUE),
      sd_unsg_vllm   = sd(unsg_vLLM, na.rm = TRUE),
      se_unsg_vllm   = sd_unsg_vllm / sqrt(n()),
      mean_unsg_edf  = mean(unsg_EDF, na.rm = TRUE),
      sd_unsg_edf    = sd(unsg_EDF, na.rm = TRUE),
      se_unsg_edf    = sd_unsg_edf / sqrt(n()),
      n              = n(),
      .groups = "drop"
    )
  
  # Réorganiser les données pour ggplot (long format)
  df_long_sla <- df_unsg_sla %>%
    pivot_longer(
      cols = c(mean_unsg_vllm, mean_unsg_edf, se_unsg_vllm, se_unsg_edf),
      names_to = c(".value", "method"),
      names_pattern = "(mean|se)_(unsg_.*)"
    ) %>%
    mutate(
      method = recode(method,
                      "unsg_vllm" = "vLLM",
                      "unsg_edf"  = "EDF")
    )
  
  
  ggplot(df_long_sla, aes(
    x = percent_urgent,
    y = mean,
    color = factor(sla_factor),
    group = interaction((sla_factor), method),
    linetype = method
  )) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(
      ymin = mean - se,
      ymax = mean + se,
      fill = factor(sla_factor)
    ), alpha = 0.15, color = NA) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    facet_grid(deadline_type ~ ., scales = "free_y", labeller = labeller(
      deadline_type = function(x) paste("Deadline type =", x)
    )) +
    scale_linetype_manual(values = c("vLLM" = "solid", "EDF" = "dotted")) +
    scale_x_continuous(breaks = seq(10, 100, by = 10), limits = c(10, 100)) +
    labs(
      title = "UNSG for Different SLA Factor vs. Urgent Requests",
      subtitle = xp_info,
      x = "Percentage of urgent requests",
      y = "Urgency-normalized success gain (%)",
      color = "SLA factor",
      fill  = "SLA factor",
      linetype = "Method"
    ) +
    theme(legend.position = "top")
  
  # 
  # ggplot(
  #   df_unsg_sla,
  #   aes(
  #     x = percent_urgent,
  #     color = factor(sla_factor),
  #     group = sla_factor
  #   )
  # ) +
  #   geom_line(aes(y = mean_unsg, linetype = "baseline"), linewidth = 1) +
  #   geom_ribbon(
  #     aes(
  #       ymin = mean_unsg - se,
  #       ymax = mean_unsg + se,
  #       fill = factor(sla_factor)
  #     ),
  #     alpha = 0.15,
  #     color = NA
  #   ) +
  #   
  #   geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  #   
  #   facet_grid(
  #     deadline_type ~ .,
  #     labeller = labeller(
  #       rt_scaling = function(x) paste("RT Scaling =", x),
  #       sla_factor = function(x) paste("SLA factor =", x),
  #       deadline_type = function(x) paste("Deadline type =", x)
  #     )
  #   ) +
  #   
  #   labs(
  #     title = "UNSG for Different SLA factor in Function of Urgent Request",
  #     subtitle = xp_info,
  #     x = "Percentage of urgent requests",
  #     y = "Urgency-normalized success gain (%)",
  #     color = "SLA Factor",
  #     fill  = "SLA Factor",
  #     linetype = "Method"
  #   ) +
  #   
  #   scale_x_continuous(
  #     breaks = seq(10, 100, by = 10),
  #     limits = c(10, 100)
  #   ) +
  #   
  #   theme(legend.position = "top")
  # 
  ggsave(paste0("pdf_results/unsg_sla", xp_suffix, ".pdf"),width=8, height=4)
  
  
  
  ############# Thgroughput
  df_throughput_rt <- df %>%
    filter(nb_req==100) %>%
    # filter(diff_success_ratio != 0)%>%
    group_by(rt_scaling,
             nb_req,
             deadline_type,
             percent_urgent,
             method) %>%
    select(rt_scaling,
           nb_req,
           deadline_type,
           percent_urgent,
           useful_throughput,
           method
    ) %>%
    summarise(
      n         = n(),
      mean_useful_throughput= mean( useful_throughput, na.rm = TRUE),
      sd   = sd( useful_throughput, na.rm = TRUE),
      se   = sd / sqrt(n),
      .groups   = "drop"
    )
  
  df_throughput_sla <- df %>%
    filter(nb_req==100) %>%
    # filter(diff_success_ratio != 0)%>%
    group_by(sla_factor,
             nb_req,
             deadline_type,
             percent_urgent,
             method) %>%
    select(sla_factor,
           nb_req,
           deadline_type,
           percent_urgent,
           useful_throughput,
           method
    ) %>%
    summarise(
      n         = n(),
      mean_useful_throughput= mean( useful_throughput, na.rm = TRUE),
      sd   = sd( useful_throughput, na.rm = TRUE),
      se   = sd / sqrt(n),
      .groups   = "drop"
    )
  
  ggplot(
    df_throughput_rt,
    aes(
      x = percent_urgent,
      y = mean_useful_throughput,
      color = method,
      group = interaction(method, rt_scaling)
    )
  ) +
    geom_line(linewidth = 1) +
    facet_grid(rt_scaling ~ deadline_type, scales = "free_y") +
    scale_x_continuous(breaks = seq(10, 100, 10), limits = c(10, 100)) +
    labs(
      title = "Useful Throughput vs Urgency for different SLA and Deadlines Distribution",
      subtitle = xp_info,
      x = "Percent urgent (%)",
      y = "Useful throughput (req/s)",
      color = "Scheduler"
    ) 
  
  
  ggsave(paste0("pdf_results/throughput_rt", xp_suffix, ".pdf"), width = 8, height = 8)
  
  ggplot(
    df_throughput_sla,
    aes(
      x = percent_urgent,
      y = mean_useful_throughput,
      color = method,
      group = interaction(method, sla_factor)
    )
  ) +
    geom_line(linewidth = 1) +
    facet_grid(sla_factor ~ deadline_type, scales = "free_y") +
    scale_x_continuous(breaks = seq(10, 100, 10), limits = c(10, 100)) +
    labs(
      title = "Useful Throughput vs Urgency for different SLA and Deadlines Distribution",
      subtitle = xp_info,
      x = "Percent urgent (%)",
      y = "Useful throughput (req/s)",
      color = "Scheduler"
    ) 
  
  ggsave(paste0("pdf_results/throughput_sla", xp_suffix, ".pdf"), width = 8, height = 8)
  
  
  ################ Percentage of Urgent request meeting their deadlines
  
  df_f <- df %>%
    filter(nb_req == 100)
  
  df_f$method
  df_summary <- df_f %>%
    group_by(percent_urgent, sla_factor, rt_scaling, deadline_type, method) %>%
    summarise(
      mean = mean(success_ratio_urgent, na.rm = TRUE),
      sd   = sd(success_ratio_urgent, na.rm = TRUE),
      n      = n(),
      .groups = "drop"
    ) %>%
    mutate(
      stderr = sd / sqrt(n),
    ) 
  
  
  ggplot(df_summary,
         aes(x = percent_urgent,
             y = mean,
             color = method,
             fill = method)) +
    geom_ribbon(
      aes(ymin = mean - stderr, ymax = mean + stderr),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_grid(
      sla_factor + deadline_type ~ rt_scaling,
      scales = "free_y",
      labeller = labeller(
        rt_scaling   = function(x) paste("RT Scaling =", x),
        sla_factor   = function(x) paste("SLA factor =", x),
        deadline_type = function(x) paste("DL type =", x)
      )
    ) +
    labs(
      # title = "Percentage of Urgent Requests Meeting Their Deadline",
      subtitle = xp_info,
      x = "Percentage of urgent requests",
      y = "Percentage of urgent requests meeting their deadline",
      color = "Scheduler",
      fill  = "Scheduler"
    )+
    theme(legend.position = "top")
  
  ggsave(paste0("pdf_results/Percentage_urgent_requests_meeting_their_deadline", xp_suffix, ".pdf"), width = 8, height = 8)
  
  
  ############# Urgent Thgroughput
  df_urgent_throughput <- df %>%
    filter(nb_req==100) %>%
    # filter(diff_success_ratio != 0)%>%
    group_by(percent_urgent,
             deadline_type,
             method
    ) %>%
    select(percent_urgent,
           deadline_type,
           useful_throughput_urgent,
           method
    ) %>%
    summarise(
      n         = n(),
      mean_useful_throughput_urgent = mean( useful_throughput_urgent, na.rm = TRUE),
      sd   = sd( useful_throughput_urgent, na.rm = TRUE),
      se   = sd / sqrt(n),
      .groups   = "drop"
    ) 
  
  
  ggplot(
    df_urgent_throughput,
    aes(
      x = percent_urgent,
      y = mean_useful_throughput_urgent,
      color = method,
      fill  = method,
      group = method
    )
  ) +
    geom_ribbon(
      aes(
        ymin = mean_useful_throughput_urgent - se,
        ymax = mean_useful_throughput_urgent + se
      ),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    facet_grid( deadline_type ~ ., scales = "free_y",
                labeller = labeller(
                  deadline_type = function(x) paste("Deadline type =", x)
                )
    ) +
    scale_x_continuous(breaks = seq(10, 100, 10),
                       limits = c(10, 100)) +
    labs(
      title="Usefull urget throuput in Function of Urgent Requests",
      subtitle = xp_info,
      x = "Percentage of urgent requests",
      y = "Goodput (tok/s)",
      color = "Scheduler",
      fill  = "Scheduler"
    ) +
    theme(legend.position = "top")
  
  ggsave(paste0("pdf_results/throughput_urgent_all", xp_suffix, ".pdf"), width = 8, height = 4)
  
  
  
  ############# Goodput
  
  df_goodput <- df %>%
    filter(nb_req==100) %>%
    # filter(diff_success_ratio != 0)%>%
    group_by(percent_urgent,
             deadline_type,
             method
    ) %>%
    select(percent_urgent,
           deadline_type,
           goodput,
           method
    ) %>%
    summarise(
      n         = n(),
      mean_goodput = mean(goodput, na.rm = TRUE),
      sd   = sd( goodput, na.rm = TRUE),
      se   = sd / sqrt(n),
      .groups   = "drop"
    ) 
  
  
  ggplot(
    df_goodput,
    aes(
      x = percent_urgent,
      y = mean_goodput,
      color = method,
      fill  = method,
      group = method
    )
  ) +
    geom_ribbon(
      aes(
        ymin = mean_goodput - se,
        ymax = mean_goodput + se
      ),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    facet_grid( deadline_type ~ ., scales = "free_y",
                labeller = labeller(
                  deadline_type = function(x) paste("Deadline type =", x)
                )
    ) +
    scale_x_continuous(breaks = seq(10, 100, 10),
                       limits = c(10, 100)) +
    labs(
      title="Goodput in Function of Urgent Requests",
      subtitle = xp_info,
      x = "Percentage of urgent requests",
      y = "Goodput (tok/s)",
      color = "Scheduler",
      fill  = "Scheduler"
    ) +
    theme(legend.position = "top")
  
  ggsave(paste0("pdf_results/Goodput_", xp_suffix, ".pdf"), width = 8, height = 4)
  
  src <- file.path(getwd(), "pdf_results")
  files <- list.files(src, full.names = TRUE)
  
  dest <- "./new_figs"
  res <- file.copy(files[file.info(files)$isdir == FALSE], dest, overwrite = TRUE)
  
  print(paste("All files copied to:", dest))
  
  df_table <- df_f %>%
    filter(deadline_type=="Det.", sla_factor==5, rt_scaling==0.1, percent_urgent>0) %>%
    group_by(method, percent_urgent) %>%
    summarise(
      mean = mean(success_ratio_urgent, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    pivot_wider(
      names_from = percent_urgent,
      values_from = mean
    ) %>%
    arrange(method)
  
  print(df_table)
  
  # Compute mean success per setting
  df_mean <- df_f %>%
    filter(percent_urgent > 0) %>%
    group_by(method, deadline_type, sla_factor, rt_scaling, percent_urgent) %>%
    summarise(mean_suc = mean(success_ratio_urgent, na.rm = TRUE), .groups = "drop") 
  
  return(df_mean)
  
  # Compute ratio against a baseline (e.g., vLLM sched. or EDF)
  baseline_method <- "vLLM sched."  # or "EDF"
  
  df_ratio <- df_mean %>%
    pivot_longer(cols = -c(method, deadline_type, sla_factor, rt_scaling),
                 names_to = "percent_urgent", values_to = "mean_suc") %>%
    group_by(deadline_type, sla_factor, rt_scaling, percent_urgent) %>%
    mutate(ratio = mean_suc / mean_suc[method == baseline_method]) %>%
    ungroup()
  
  # Compute geometric mean of the ratios for each method
  df_geo_mean <- df_ratio %>%
    group_by(method) %>%
    summarise(geo_mean_ratio = exp(mean(log(ratio[method != baseline_method]), na.rm = TRUE)),
              .groups = "drop")
  
  df_geo_mean
  
  
}

setwd("./input/serving_request/emulation_results_openai")
df_mean <- generate_figs("results_deadline_mistral_nvidia-a100-80gb-pcie_4.csv")



p_u_r_df<- function(file, gpu){
  df <- read.csv(file)
  df <- df %>% 
    filter(rt_scaling %in% c(1e-5, 0.001, 0.1)) %>% 
    filter(nb_req == 100) %>%
    filter(skip_lines %in% c(
      0, 50000, 100000, 150000, 200000, 250000, 300000, 350000,
      400000, 450000, 500000, 550000, 600000, 650000, 700000,
      750000, 800000, 850000, 900000, 950000, 1000000
    )) %>%
    filter(trace=="BurstGPT")
  
  
  df <- df %>%
    mutate(
      deadline_type = recode(
        deadline_type,
        "det-deadlines" = "Det.",
        "rnd-deadlines" = "Rnd"
      )
    ) 
  
  df$method <- dplyr::recode(df$method,
                             "baseline" = "vLLM sched.",
                             "out-of-order-discard-most-urgent" = "Deadline-aware",
                             "out-of-order-edf" = "EDF"
  )
  # Force the plotting order
  df$method <- factor(df$method, levels = c(
    "vLLM sched.",
    "Deadline-aware",
    "EDF"
  ))
  
  

df_f <- df %>%
  filter(percent_urgent>0)

df_summary <- df_f %>%
  group_by(percent_urgent, sla_factor, rt_scaling, deadline_type, method) %>%
  summarise(
    mean = mean(success_ratio_urgent, na.rm = TRUE),
    sd   = sd(success_ratio_urgent, na.rm = TRUE),
    n      = n(),
    .groups = "drop"
  ) %>%
  mutate(
    stderr = sd / sqrt(n),
  ) 

  df_summary$gpu <- gpu
  return(df_summary)
}

plot_p_u_r <- function(df){
  df<-df_qwen
  g<-ggplot(df,
         aes(x = percent_urgent,
             y = mean,
             color = method,
             fill = method)) +
    geom_ribbon(
      aes(ymin = mean - stderr, ymax = mean + stderr),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_grid(
      gpu+sla_factor + deadline_type ~ rt_scaling,
      scales = "free_y",
      labeller = labeller(
        rt_scaling   = function(x) paste("RT Scaling =", x),
        sla_factor   = function(x) paste("SLA =", x),
        deadline_type = function(x) paste("DL =", x)
      )
    ) +
    labs(
      # title = "Percentage of Urgent Requests Meeting Their Deadline",
      subtitle = "QWEN-30B - 1 H1000 94GB or 4xA100 80GB",
      x = "Percentage of urgent requests",
      y = "Percentage of urgent requests meeting their deadline",
      color = "Scheduler",
      fill  = "Scheduler"
    )+
    theme(legend.position = "top")
  
  g
  ggsave("pdf_results/Percentage_urgent_requests_meeting_their_deadline_QWEN30B.pdf", width = 8, height = 10)
  
  
  dest <- "./new_figs"
  res <- file.copy("pdf_results/Percentage_urgent_requests_meeting_their_deadline_QWEN30B.pdf", dest, overwrite = TRUE)
  
  return(g)
  }


 setwd("./input/serving_request/emulation_results_openai")
 df_A100 <- p_u_r_df("results_deadline_qwen_nvidia-a100-80gb-pcie_4.csv","4xA100")
 
 
 setwd("./input/serving_request/refactoring/emulation_results_openai/")
 df_H100<- p_u_r_df("results_deadline_qwen_nvidia-h100-nvl_1.csv","1xH100")
 
 df_qwen <-bind_rows(df_A100,df_H100) %>%
   filter(sla_factor>2)
 
 g <- plot_p_u_r(df_qwen)
 g
 
 setwd("./input/serving_request/emulation_results_openai")
 generate_figs("results_deadline_mistral_nvidia-a100-80gb-pcie_1.csv")
 generate_figs("results_deadline_mistral_nvidia-a100-80gb-pcie_4.csv")
 generate_figs("results_deadline_qwen_nvidia-a100-80gb-pcie_4.csv")
 
 setwd("./input/serving_request/refactoring/emulation_results_openai/")
 generate_figs("results_deadline_qwen_nvidia-h100-nvl_1.csv")
 
setwd("./input/serving_request/emulation_results_with_percent")
generate_figs("results_deadline_mistral_nvidia-h100-nvl_1.csv")



