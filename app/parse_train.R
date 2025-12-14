library(tidyverse)
library(readr)
library(readxl)
library(rlang)
library(lubridate)

dat <- read_excel("~/Documents/hackathon-upc/data/IQ_Cancer_Endometrio_merged_NMSP.xlsx", na = 'NA', sheet = 'IQ_Cancer_Endometrio_merged_NMS')
dat



# Parsing functions
date_to_days <- function(x) {
  x <- as.character(x)
  out <- rep(NA_real_, length(x))
  
  # dd/mm/yyyy
  idx1 <- grepl("^\\d{2}/\\d{2}/\\d{4}$", x)
  out[idx1] <- as.numeric(
    as.Date(x[idx1], format = "%d/%m/%Y")
  )
  
  # yyyy-dd-mm
  idx2 <- grepl("^\\d{4}-\\d{2}-\\d{2}$", x)
  out[idx2] <- as.numeric(
    as.Date(x[idx2], format = "%Y-%d-%m")
  )

  out
}


percent_to_unit <- function(x) {
  x <- as.numeric(x)
  x[x < 0 | x > 100] <- NA
  x / 100
}


binary_with_na <- function(x, na_codes=c()) {
  x <- as.numeric(x)
  x[x %in% na_codes] <- NA
  
  ux <- sort(unique(x[!is.na(x)]))
  if (length(ux) != 2)
    stop("Binary variable does not have exactly 2 valid values")
  
  as.integer(x == ux[2])
}


categorical_auto <- function(x, prefix, na_codes=c()) {
  x <- as.numeric(x)
  x[x %in% na_codes] <- NA
  
  levels <- sort(unique(x[!is.na(x)]))
  
  # If 2 categories treat as binary
  if (length(levels) == 2) {
    return(tibble(!!prefix := as.integer(x == levels[2])))
  }
  
  # One-hot (exclude first level)
  dummies <- lapply(levels[-1], function(lv) {
    out <- as.integer(x == lv)
    out[is.na(x)] <- NA_integer_
    out
  })
  names(dummies) <- paste0(prefix, "_", levels[-1])
  as_tibble(dummies)
}


important_cols <- c(
  'FN',   # Date dd/mm/yyyy
  'edad',   # Years (float)
  'imc',   # float
  'f_diag',   # Date yyyy-dd-mm
  'fecha_qx',   # Date dd/mm/yyyy
  'tipo_histologico',   # Category (int 1-12, 88) 88: otros
  'Grado',   # Category (1, 2)
  'valor_de_ca125',   # float
  'ecotv_infiltobj',   # 1: no aplicado, 2: <50%, 3: >50%, 4: no valorable
  'ecotv_infiltsub',   # 1: no aplicado, 2: <50%, 3: >50%, 4: no valorable
  'infiltracion_mi',   # Category (0, 1, 2, 3)
  'infilt_estr_cervix',   # Binary (0, 1, 2) 2: se desconoce
  'metasta_distan',   # Binary (0, 1)
  'grupo_riesgo',   # Category (0, 1, 2)
  'estadiaje_pre_i',   # Category (0, 1, 2)
  'tto_NA',   # Binary (0, 1)
  'tto_1_quirugico',   # Binary (0, 1)
  'asa',   # Category (0-6) 6: desconocido
  'histo_defin',   # Category (0-9) 9: otros
  'grado_histologi',   # Category (1-2)
  'tamano_tumoral',   # cm (float)
  'afectacion_linf',   # Binary (0, 1)
  'AP_centinela_pelvico',   # Category (0-4)
  'AP_ganPelv',   # Category (0-3)
  'AP_glanPaor',   # Category (0-3)
  'recep_est_porcent',   # Percentage (int 0-100)
  'rece_de_Ppor',   # Percentage (int 0-100)
  'beta_cateninap',   # Binary (0, 1, 2) 2: no realizado
  'estadificacion_',   # Category (1-9)
  'FIGO2023',   # Category (1-14)
  'grupo_de_riesgo_definitivo',   # Category (1-5) 5: avanzados
  'Tributaria_a_Radioterapia',   # Binary (0, 1)
  'bqt',   # Binary (0, 1)
  'qt',   # Binary (0, 1)
  'Tratamiento_sistemico_realizad',  # Category (0, 1, 2) 0: no realizado, 1: dosis parcial, 2: dosis completa
  'Tratamiento_sistemico',   # Binary (0, 1)
  'visita_control',   # Date dd/mm/yyyy
  'recidiva',   # Binary (0, 1, 2) 2: desconocido
  'est_pcte',    # Binary (1, 2, 3) 1: viva, 2: muerta, 3: desconocido
  'causa_muerte',   # Binary (0, 1)
  'f_muerte',   # Date dd/mm/yyyy
  'libre_enferm',   # Binary (0, 1, 2) 2: desconocido
  'numero_de_recid',   # Int (>=0)
  'fecha_de_recidi',   # Date dd/mm/yyyy
  'dx_recidiva',   # Category (0, 1)
  'tto_recidiva',   # Category (0, 1, 2)
  'Tt_recidiva_qx',   # Category (0, 1, 2, 3)
  'Reseccion_macroscopica_complet'   # Binary (0, 1)
)

one_hot_patterns <- c(
  "^loc_recidiva_r\\d+$"   # Category (1-6) 6: no realizado
# "^estudio_genetico_r\\d+$"   # Category (1-6)   4: no consta
)


transform_map <- list(
  # Dates
  FN = ~date_to_days(.x),
  f_diag = ~date_to_days(.x),
  visita_control = ~date_to_days(.x),
  f_muerte = ~date_to_days(.x),
  fecha_de_recidi = ~date_to_days(.x),
  fecha_qx = ~date_to_days(.x),
  
  # Continuous
  edad = ~as.numeric(.x),
  imc = ~as.numeric(.x),
  valor_de_ca125 = ~as.numeric(.x),
  tamano_tumoral = ~as.numeric(.x),
  numero_de_recid = ~as.integer(.x),
  
  # Percentages
  recep_est_porcent = ~percent_to_unit(.x),
  rece_de_Ppor = ~percent_to_unit(.x),
  
  # Binary (+ NA)
  metasta_distan = ~binary_with_na(.x),
  tto_NA = ~binary_with_na(.x),
  tto_1_quirugico = ~binary_with_na(.x),
  afectacion_linf = ~binary_with_na(.x),
  Tributaria_a_Radioterapia = ~binary_with_na(.x),
  bqt = ~binary_with_na(.x),
  qt = ~binary_with_na(.x),
  Tratamiento_sistemico_realizad = ~binary_with_na(.x, c(0)),
  Tratamiento_sistemico = ~binary_with_na(.x),
  causa_muerte = ~binary_with_na(.x),
  Reseccion_macroscopica_complet = ~binary_with_na(.x),
  infilt_estr_cervix = ~binary_with_na(.x, c(2)),
  recidiva = ~binary_with_na(.x, c(2)),
  est_pcte = ~binary_with_na(.x, c(3)),
  Grado = ~binary_with_na(.x)
)

categorical_cols <- list(
  libre_enferm = c(),
  beta_cateninap = c(),
  infiltracion_mi = c(),
  tipo_histologico = c(),
  ecotv_infiltobj = c(),
  ecotv_infiltsub = c(),
  grupo_riesgo = c(),
  estadiaje_pre_i = c(),
  asa = c(6),
  histo_defin = c(),
  grado_histologi = c(),
  AP_centinela_pelvico = c(),
  AP_ganPelv = c(),
  AP_glanPaor = c(),
  estadificacion_ = c(),
  FIGO2023 = c(),
  grupo_de_riesgo_definitivo = c(),
  dx_recidiva = c(),
  tto_recidiva = c(),
  Tt_recidiva_qx = c()
)





dat_important <- dat %>% select(all_of(important_cols), matches(paste(one_hot_patterns, collapse = "|")))

# Pre-parsing inputation
dat_important <- dat_important %>%
  mutate(estadificacion_ = case_when(!is.na(estadificacion_) ~ estadificacion_,
                                     estadiaje_pre_i == 0 ~ 1,
                                     estadiaje_pre_i == 1 ~ 3,
                                     estadiaje_pre_i == 2 ~ 7,
                                     TRUE ~ estadificacion_),
         FIGO2023 = case_when(!is.na(FIGO2023) ~ FIGO2023,
                               estadiaje_pre_i == 0 ~ 1,
                               estadiaje_pre_i == 1 ~ 7,
                               estadiaje_pre_i == 2 ~ 11,
                               TRUE ~ FIGO2023))

# One-hot encoding
dat_clean <- dat_important %>%
  select(-all_of(names(categorical_cols))) %>% 
  bind_cols(
    lapply(names(categorical_cols), function(nm) {
      categorical_auto(
        dat_important[[nm]],
        prefix = nm,
        na_codes = categorical_cols[[nm]]
      )
    }) %>% bind_cols()
  ) %>%
  mutate(
    across(
      matches(paste(one_hot_patterns, collapse = "|")),
      ~as.integer(.x)
    )
  )


# Function transformation
for (nm in names(transform_map)) {
  print(nm)
  dat_clean[[nm]] <- as_function(transform_map[[nm]])(
    dat_clean[[nm]]
  )
}


# Times and outcome calculation
dat_clean <- dat_clean  %>%
  mutate(os_date = ifelse(is.na(f_muerte), visita_control, f_muerte)) %>%
  mutate(dfs_date = ifelse(is.na(fecha_de_recidi), visita_control, fecha_de_recidi)) %>%
  mutate(dfs_time = dfs_date - fecha_qx) %>%
  mutate(os_time = os_date - fecha_qx) %>%
  mutate(dfs_status = ifelse(recidiva == 1, 1, 0)) %>%
  mutate(os_status = ifelse(est_pcte == 1, est_pcte, 0)) %>%
  mutate(risk_status = ifelse((dfs_time/365) < 3, ifelse(dfs_status == 1, 1, NA), 0))

dat_clean <- dat_clean %>%
  filter(dfs_time >= 0 | is.na(dfs_time))

# Post-parsing inputing
dat_clean <- dat_clean %>%
  mutate(tamano_tumoral = if_else(is.na(tamano_tumoral), round(mean(tamano_tumoral, na.rm = TRUE), 1), tamano_tumoral))



# File writing
dat_clean %>% write_tsv('data_clean.tsv')


dat_clean %>%
  select(-c(FN, dfs_date, f_diag, f_muerte, fecha_qx, fecha_de_recidi, os_time, os_date)) %>%
  select(-c(recidiva, starts_with('tto_recidiva'), starts_with('tt_recidiva_qx'), starts_with('loc_recidiva'),
            dx_recidiva, numero_de_recid, causa_muerte, visita_control, est_pcte, starts_with('libre_enferm'))) %>%
  write_tsv('data_train.tsv')
