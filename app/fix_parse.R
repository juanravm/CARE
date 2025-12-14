library(tidyverse)
library(readr)
library(readxl)
library(rlang)
library(lubridate)



dat <- args$df

# Parsing functions
numeric_fixed <- function(x, nm, default){
  out <- as.numeric(x)
  
  if (any(is.na(out))) {
    print(paste('WARNING! Missing values for', nm, 'will be replace by mean of the training set:', default))
    out[is.na(out)] <- default
  }

  out
}
binary_fixed <- function(x, nm, mapping, default) {
  out <- as.numeric(x)
  
  out[!as.character(out) %in% names(mapping)] <- NA
  if (any(is.na(out))) {
    print(paste('WARNING! Missing values for', nm, 'will be replace by mode of the training set:', default))
    out[is.na(out)] <- default
  }

  out <- mapping[as.character(out)]
  
  # ux <- sort(unique(out[!is.na(out)]))
  # if (length(ux) != 2)
  #   stop(paste('Variable does not have exactly 2 valid values'))
  
  ux <- sort(unique(out[!is.na(out)]))
  if (length(ux) != 2) {
    warning(
      paste(
        "Variable", nm,
        "does not have exactly 2 valid values.",
        "Observed:", paste(ux, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  as.integer(out)
}


categorical_fixed <- function(x, prefix, lvls, default) {
  x <- as.numeric(x)
  
  # One-hot (exclude first level)
  dummies <- lapply(lvls[-1], function(lv) {
    as.integer(x == lv)
  })
  
  names(dummies) <- paste0(prefix, "_", lvls[-1])
  as_tibble(dummies)
}


direct_cols <-  list(
  edad = ~numeric_fixed(.x, 'edad', 62.2),   # Years (float)
  imc = ~numeric_fixed(.x, 'imc', 30.8),   # float
  Grado = ~binary_fixed(.x, 'Grado', c(`1`=0, `2`=1), 1),   # Category (1, 2)
  infilt_estr_cervix = ~binary_fixed(.x, 'infilt_estr_cervix', c(`0`=0, `1`=1), 0),   # Binary (0, 1, 2) 2: se desconoce
  metasta_distan = ~binary_fixed(.x, 'metasta_distan', c(`0`=0, `1`=1), 0),   # Binary (0, 1)
  tto_NA = ~binary_fixed(.x, 'tto_NA', c(`0`=0, `1`=1), 0),   # Binary (0, 1)
  tto_1_quirugico = ~binary_fixed(.x, 'tto_1_quirugico', c(`0`=0, `1`=1), 1),   # Binary (0, 1)
  tamano_tumoral = ~numeric_fixed(.x, 'tamano_tumoral', 3.82),   # cm (float)
  Tributaria_a_Radioterapia = ~binary_fixed(.x, 'Tributaria_a_Radioterapia', c(`0`=0, `1`=1), 0),   # Binary (0, 1)
  bqt = ~binary_fixed(.x, 'bqt', c(`0`=0, `1`=1), 0),   # Binary (0, 1)
  qt = ~binary_fixed(.x, 'qt', c(`0`=0, `1`=1), 0)   # Binary (0, 1)
)

one_hot_cols <- list(
  beta_cateninap = list(c(0, 1, 2), 0),   # Binary (0, 1, 2) 2: no realizado
  tipo_histologico = list(c(1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 88), 2),   # Category (int 1-12, 88) 88: otros
  ecotv_infiltobj = list(c(1, 2, 3, 4), 4),   # 1: no aplicado, 2: <50%, 3: >50%, 4: no valorable
  ecotv_infiltsub = list(c(1, 2, 3, 4), 2),   # 1: no aplicado, 2: <50%, 3: >50%, 4: no valorable
  grupo_riesgo = list(c(1, 2, 3), 1),   # Category (0, 1, 2)
  estadiaje_pre_i = list(c(0, 1, 2), 0),   # Category (0, 1, 2)
  histo_defin = list(c(1, 2, 3, 4, 5, 6, 7, 8, 9), 2),   # Category (1-9) 9: otros
  estadificacion_ = list(c(1, 2, 3, 4, 5, 6, 7, 8, 9), 1),   # Category (1-9)
  FIGO2023 = list(c(1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14), 1)   # Category (1-14)
)

available_direct <- intersect(names(direct_cols), colnames(dat))
available_one_hot <- intersect(names(one_hot_cols), colnames(dat))
missing_direct <- setdiff(names(direct_cols), colnames(dat))
missing_one_hot <- setdiff(names(one_hot_cols), colnames(dat))

if (length(missing_direct) > 0) {
  message(
    paste(
      "WARNING! Missing direct variables:",
      paste(missing_direct, collapse = ", ")
    )
  )
}

if (length(missing_one_hot) > 0) {
  message(
    paste(
      "WARNING! Missing one-hot variables:",
      paste(missing_one_hot, collapse = ", ")
    )
  )
}


# Parse!
# dat_test <- dat %>% select(all_of(names(direct_cols)), all_of(names(one_hot_cols)))
dat %>% select(all_of(available_direct), all_of(available_one_hot))

# for (nm in names(direct_cols)) {
#   dat_test[[nm]] <- as_function(direct_cols[[nm]])(dat_test[[nm]])
# }

# dat_test <- dat_test %>%
#   select(-all_of(names(one_hot_cols))) %>% 
#   bind_cols(
#     lapply(names(one_hot_cols), function(nm) {
#       categorical_fixed(
#         dat_test[[nm]],
#         prefix = nm,
#         lvls = one_hot_cols[[nm]][[1]],
#         default = one_hot_cols[[nm]][[2]]
#       )
#     }) %>% bind_cols()
#   )


dat %>%
  write_tsv(args$out_filename)
