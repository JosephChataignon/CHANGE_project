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
our education dataset and generates a question about it.
TODO: add document reference to the synthetic data export.
TODO: run the synthetic data script to make a few hundred samples.

### II.B. Evaluate search on synthetic data

We will take the synthetic question-document pairs, and submit the question to the FRAG search API.
We then check whether the original document is within the retrieved sources, and at which rank.
We should measure top-1, top-5 and top-10 matches. We will run this with various fine-tuned and 
non-fine-tuned models.
TODO: write script

## III. Qualitative evaluation

### III.A. Prepare standard questions

The experts will have 10 fixed questions and 10 "free" questions to ask the search-powered chatbot.
TODO: write thse 10 questions, with information retrieval in mind.

### III.B. Prepare FRAG app

The FRAG interface needs to be adapted for evaluation by the experts. We'll need experts to be able 
to access the app from outside the university network, and they should have a side-by-side view of 
the results with 2 different embedding models (the fine-tuned one vs. the non-fine-tuned one).
TODO: harden server, password-protect it.
TODO: chose which 2 embedding models for the blind-testing
TODO: edit FRAG app to have side-by-side view
