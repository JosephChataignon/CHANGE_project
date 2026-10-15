# FRAG Evaluation plan
round 2

## Purpose

This evaluation is intended to be used in a paper. The question we're trying to answer is: does
the fine-tuning of embedding models on a dataset improve semantic search over that dataset ?

## Setup

To answer this question we use the FRAG application as a search engine, and our dataset of documents
related to education science.
We will run both a quantitative evaluation with synthetic search data, and a qualitative evaluation 
by experts of the field

## I. Fine-tune embedding models

Already done. Fine-tuned with same-document-is closer policy.
TODO: fine-tune with synthetic question-answer pairs for another paper.

## II. Quantitative evaluation

### II.A. Generate synthetic data

The code for synthetic data generation is ready, in `synthetic_data_gen.py`. It takes a document from
our education dataset and generates a question about it. We sample larger chunks (now 24 sentences
per chunk - ingestion was 20 sentences - 3 overlap) to get more context.
We do NOT export a document reference: matching a retrieved source back to the original will be done
by text overlap against the stored chunk (the `relevant_sentence` column), not by document id.
TODO: run the synthetic data script to make a few hundred samples.
TODO: make sure synthetic questions are the right level of specificity (currently a bit vague). Maybe add context of the books. -seems better with qwen3.8

### II.B. Evaluate search on synthetic data

We will take the synthetic question-document pairs, and submit the question to the FRAG search API.
We then check whether the original chunk is among the retrieved sources, and at which rank. Because
the documents returned by FRAG may be chunked differently from how we chunked them when generating
synthetic data, we match by text overlap between each retrieved source and the original
`relevant_sentence` rather than by document reference.
We'll make a distribution plot of overlap between retrieved and original, then chose the threshold for when overlap counts as the same document. 
We should measure top-1, top-5 and top-10 matches. We will run this with various fine-tuned and 
non-fine-tuned models.
TODO: test script
TODO: use distribution plot to chose overlap threshold
TODO: Run script with various embedding models loaded, finetuned vs non-finetuned - deduct which to use for human eval

## III. Qualitative evaluation

### III.A. Prepare standard questions

~~The experts will have 10 fixed questions and 10 "free" questions to ask the search-powered chatbot.~~
~~TODO: write thse 10 questions, with information retrieval in mind.~~
In fact we'll reuse the 36 questions from validation_stage_1.

### ~~III.B. Prepare FRAG app~~

~~The FRAG interface needs to be adapted for evaluation by the experts. We'll need experts to be able ~~
~~to access the app from outside the university network, and they should have a side-by-side view of ~~
~~the results with 2 different embedding models (the fine-tuned one vs. the non-fine-tuned one).~~
~~TODO: harden server, password-protect it.~~
~~TODO: chose which 2 embedding models for the blind-testing~~
~~TODO: edit FRAG app to have side-by-side view~~

### III.C. Run questionnaire through FRAG

Run all the questions through the FRAG system, save the answers to CSV. Run all 36 questions and just 2 
answers, one from system with base model, and the other with fine-tuned model. Use the best-performing 
models from quantitative analysis.

### III.D. Make survey

Convert the CSV of questions with answers into a LimeSurvey questionnaire.

### III.E. Run survey

Submit the survey to the experts, collect answers.
TODO: make sure limesurvey is installed and accessible from outside the University.
TODO: upload survey and test that it works
TODO: gather list of experts with André, see with Tobias if we should reward them.
TODO: send the survey link

### III.F. Analyze results

Collect results and analyze them.
