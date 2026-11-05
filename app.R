library(shiny)
library(tidytext)
library(topicmodels)
library(LDAvis)
library(tm)
library(dplyr)
library(stringr)
library(rsconnect)

states_dir <- "us_states_data"
txt_files <- list.files(states_dir, pattern = "\\.txt$", full.names = TRUE)

all_states <- lapply(txt_files, function(filepath) {
  suppressWarnings(
    paste(readLines(filepath, encoding = "UTF-8"), collapse = "\n")
  )
})
names(all_states) <- tools::file_path_sans_ext(basename(txt_files))

states_df <- data.frame(
  state = names(all_states),
  text  = unlist(all_states, use.names = FALSE),
  stringsAsFactors = FALSE
)

extract_history <- function(text) {
  lines <- str_split(text, "\n")[[1]]
  start <- which(str_trim(lines) == "History")
  end   <- which(str_trim(lines) == "Geography")
  if (length(start) == 0 || length(end) == 0) return(NA)
  hist_lines <- lines[(start[1] + 1):(end[1] - 1)]
  paste(hist_lines[hist_lines != ""], collapse = " ")
}

history_df <- states_df %>%
  mutate(history_text = sapply(text, extract_history)) %>%
  filter(!is.na(history_text))

custom_stops <- data.frame(word = c(
  "state", "states", "united", "american", "also", "known",
  "located", "including", "area", "city", "cities", "town",
  "county", "region", "large", "small", "total", "number",
  "part", "new", "old", "major", "local", "national", "federal",
  "population", "census", "wikipedia", "retrieved", "isbn",
  "alabama", "alaska", "arizona", "arkansas", "california",
  "colorado", "connecticut", "delaware", "florida", "georgia",
  "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
  "kentucky", "louisiana", "maine", "maryland", "massachusetts",
  "michigan", "minnesota", "mississippi", "missouri", "montana",
  "nebraska", "nevada", "hampshire", "jersey", "mexico", "york",
  "carolina", "dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
  "rhode", "tennessee", "texas", "utah", "vermont", "virginia",
  "washington", "wisconsin", "wyoming"
))

ui <- fluidPage(
  titlePanel("US States Historical Topic Modeling"),
  
  sidebarLayout(
    sidebarPanel(
      sliderInput("k", 
                  "Number of Topics:", 
                  min = 3, max = 12, value = 6, step = 1),
      
      sliderInput("tfidf_threshold",
                  "TF-IDF Threshold (higher = more distinctive words):",
                  min = 0.5, max = 0.95, value = 0.75, step = 0.05),
      
      actionButton("run", "Run Model", 
                   class = "btn-primary", 
                   style = "width: 100%; margin-top: 10px;"),
      
      hr(),
      helpText("Adjust the number of topics and TF-IDF threshold, then click Run Model."),
      helpText("Higher TF-IDF = fewer, more distinctive words per state."),
      
      hr(),
      h4("States included:"),
      verbatimTextOutput("state_count")
    ),
    
    mainPanel(
      htmlOutput("ldavis_plot")
    )
  )
)

# ── 3. Server ─────────────────────────────────────────────────
server <- function(input, output, session) {
  
  output$state_count <- renderText({
    paste(nrow(history_df), "of 50 states")
  })
  
  lda_result <- eventReactive(input$run, {
    
    withProgress(message = "Running topic model...", value = 0, {
      
      incProgress(0.2, detail = "Tokenizing text...")
      tidy_history <- history_df %>%
        select(state, history_text) %>%
        rename(text = history_text) %>%
        unnest_tokens(word, text) %>%
        anti_join(stop_words, by = "word") %>%
        anti_join(custom_stops, by = "word") %>%
        filter(!str_detect(word, "[0-9]")) %>%
        count(state, word) %>%
        bind_tf_idf(word, state, n) %>%
        filter(tf_idf > quantile(tf_idf, input$tfidf_threshold))
      
      incProgress(0.3, detail = "Building document-term matrix...")
      dtm_history <- tidy_history %>%
        cast_dtm(state, word, n)
      
      incProgress(0.3, detail = "Fitting LDA model...")
      lda_model <- LDA(dtm_history, k = input$k, control = list(seed = 42))
      
      incProgress(0.2, detail = "Preparing visualization...")
      lda_json <- createJSON(
        phi            = posterior(lda_model)$terms,
        theta          = posterior(lda_model)$topics,
        doc.length     = rowSums(as.matrix(dtm_history)),
        vocab          = colnames(dtm_history),
        term.frequency = colSums(as.matrix(dtm_history))
      )
    })
    
    lda_json
  })
  
  output$ldavis_plot <- renderUI({
    req(lda_result())
    
    tmp_dir <- file.path(tempdir(), "ldavis")
    serVis(lda_result(), out.dir = tmp_dir, open.browser = FALSE)
    
    addResourcePath("ldavis", tmp_dir)
    tags$iframe(
      src    = "ldavis/index.html",
      width  = "100%",
      height = "700px",
      frameborder = "0"
    )
  })
}

shinyApp(ui = ui, server = server)