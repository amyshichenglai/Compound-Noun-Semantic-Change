# run on 2024-12-12
./scripts/create_and_cache_examples.sh --output_dir 2024-12-12-cached-examples-with-hyphens/ --model domain-adapted-models/COHA/2023-05-22-random-access-domain-adapt/ --test_file cordeiro-ratings-onlyNN.txt --test_file_type cordeiro --device cuda:3 --data cached_inputs/COHA/COHA_all_cased_lemmatized.pickle --lang en --ordinal 2nd-order --related_mods 2024-04-08-related-compound-candidates/related-compond-candidates-COHA-mods-counts-by-era-min-count-10.tsv --related_heads 2024-04-08-related-compound-candidates/related-compond-candidates-COHA-heads-counts-by-era-min-count-10.tsv

# run on 2024-12-18:
./scripts/create_and_cache_examples.sh --output_dir 2024-12-12-cached-examples-with-hyphens-EN/ --model domain-adapted-models/COHA/2023-05-22-random-access-domain-adapt/ --test_file cordeiro-ratings-onlyNN-OCCURRING-IN-COHA.txt  --test_file_type cordeiro --device cuda:2 --data cached_inputs/COHA/COHA_all_cased_lemmatized.pickle --lang en --ordinal 2nd-order --related_mods 2024-04-08-related-compound-candidates/related-compond-candidates-COHA-mods-counts-by-era-min-count-10.tsv --related_heads 2024-04-08-related-compound-candidates/related-compond-candidates-COHA-heads-counts-by-era-min-count-10.tsv

note that this is not bothering to cache the uses of e.g. 'video game'
