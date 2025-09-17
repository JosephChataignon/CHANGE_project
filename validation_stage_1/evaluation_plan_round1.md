# FRAG Evaluation plan

## I. Get benchmark questions

Done. They were written by André some time ago already.

## II. Run those questions through FRAG and LLMs

The questions will be sent automatically to FRAG or to simple LLMs and their answers will be stored. Various options are possible, here are the main axes where we'd need comparisons:
- with/without documents: to know whether it is any use to have RAG at all
- open/closed-source models: can we hold the comparison with commercial models if we keep everything local (TODO: get API keys for commercial models)
- non-/reasoning models: is chain-of-thought usefulin this case
- big/small models: by how much does size improve performance
That would be 16 setups to evaluate if we do every possible combination. If we drop one comparison it's only 8.
TODO: select the setups to run

## III. Make experts evaluate answers

Gather experts. TODO: decide where, when, who ?

At this point I will have a big spreadsheet with one row per question (36 questions) and one column per setup as defined in II. The columns also comprise Domain, Complexity Level and Aspect.

We will randomly split and assign questions to the experts, and then shuffle the answers. Model names will be obfuscated to avoid bias. TODO: decide how many questions and answers will each expert evaluate.

Experts will only receive a smaller spreadsheet with their assigned questions and answers. They will have one column to give a grade from 1 to 5 for each answer, and one column for qualitative remarks.
TODO: confirm that these elements are ok:
- do not include documents in the sheet ?
- we evaluate with a grade from 1-5, rather than ranking or comparison or yes/no
- for "thinking" models, cut out the thinking process
At the end we collect the spreadsheets and merge the results



