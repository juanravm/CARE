.libPaths(c(.libPaths(), "/home/natasamortvanski@vhio.org/miniconda3/envs/env_work_1/lib/R/library"))

# app.R
# A minimal Shiny app with a linear regression model,
# user inputs, prediction, stratification, and plotting

# -----------------------------
# 1. Load required libraries
# -----------------------------
library(shiny)
library(ggplot2)
library(tidyverse)
library(bslib)
library(thematic)
# -----------------------------
# 2. Create a simple example dataset
# -----------------------------
set.seed(123)

n <- 100
example_data <- data.frame(
  x1 = rnorm(n, mean = 50, sd = 10),
  x2 = rnorm(n, mean = 0, sd = 1)
)

# True underlying relationship + noise
example_data$score <- 0.5 * example_data$x1 + 2 * example_data$x2 + rnorm(n, sd = 5)

# Median-based stratification
median_score <- median(example_data$score)
example_data$group <- ifelse(example_data$score >= median_score, "High", "Low")

# -----------------------------
# 3. Fit a linear regression model
# -----------------------------
lin_model <- lm(score ~ x1 + x2, data = example_data)

# -----------------------------
# 4. Define UI
# -----------------------------
ui <- page_sidebar(
  title = "Clinical Assessment of Risk in Endometrial Cancer (CARE)",
  sidebar = sidebar(
    numericInput(
      inputId = "input_x1",
      label = "Input variable x1",
      value = 50
    ),
    numericInput(
      inputId = "input_x2",
      label = "Input variable x2",
      value = 0
    )
  ),

  mainPanel(
    plotOutput("scatterPlot"),
    hr(),
    textOutput("predictionText"),
    textOutput("groupText")
  )
)



# -----------------------------
# 5. Define server logic
# -----------------------------
server <- function(input, output) {

  # Reactive prediction for the user-defined inputs
  predicted_score <- reactive({
    new_data <- data.frame(
      x1 = input$input_x1,
      x2 = input$input_x2
    )
    predict(lin_model, newdata = new_data)
  })

  # Plot with highlighted user point
  output$scatterPlot <- renderPlot({
    user_score <- predicted_score()

    plot_data <- example_data

    ggplot(plot_data, aes(x = x1, y = score, color = group)) +
      geom_point(alpha = 0.6) +
      geom_point(
        aes(x = input$input_x1, y = user_score),
        color = "black",
        size = 4
      ) +
      geom_hline(yintercept = median_score, linetype = "dashed") +
      labs(
        title = "Linear regression dataset with user prediction",
        x = "x1",
        y = "Score"
      ) +
      theme_minimal()
  })

  # Text output: predicted score
  output$predictionText <- renderText({
    paste0("Predicted score: ", round(predicted_score(), 2))
  })

  # Text output: stratification group
  output$groupText <- renderText({
    group <- ifelse(predicted_score() >= median_score, "High", "Low")
    paste0("Predicted group: ", group)
  })
}

# -----------------------------
# 6. Run the application
# -----------------------------
shinyApp(ui = ui, server = server)
