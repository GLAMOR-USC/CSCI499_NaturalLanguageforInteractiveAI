import gensim
import tqdm


def downstream_validation(word_vectors_fn, external_val_analogies):
    print("Loading vectors from file '%s'..." % word_vectors_fn)
    wv = gensim.models.KeyedVectors.load_word2vec_format(
        word_vectors_fn, binary=word_vectors_fn[-4:] == ".bin"
    )
    print("... done loading vectors")

    print(
        "Evaluating downstream performance on analogy task over %d analogies..."
        % len(external_val_analogies)
    )
    all_correct = 0
    all_exact = 0
    all_tested = 0
    r_correct = {}
    r_exact = {}
    r_tested = {}
    t_correct = {}
    t_exact = {}
    t_tested = {}
    for t, r, abcd in tqdm.tqdm(external_val_analogies):
        if t not in t_correct:
            t_correct[t] = 0
            t_tested[t] = 0
            t_exact[t] = 0
        if (t, r) not in r_correct:
            r_correct[(t, r)] = 0
            r_tested[(t, r)] = 0
            r_exact[(t, r)] = 0
        a, b, c, d = abcd
        try:
            # A - B = C - D => D = C + B - A
            result = wv.most_similar(positive=[b, c], negative=[a], topn=1000)
            topn_words = [r[0] for r in result]
        except KeyError as err:  # word not in vocabulary; only possible when loading external word vectors
            topn_words = [None]
            print("WARNING: KeyError: {0}".format(err))
        if d in topn_words:
            t_correct[t] += 1 / (topn_words.index(d) + 1)  # reciprocol rank score
            r_correct[(t, r)] += 1 / (topn_words.index(d) + 1)  # reciprocol rank score
            all_correct += 1 / (topn_words.index(d) + 1)
        if d == topn_words[0]:
            t_exact[t] += 1
            r_exact[(t, r)] += 1
            all_exact += 1
        t_tested[t] += 1
        r_tested[(t, r)] += 1
        all_tested += 1
    print(
        "...Total performance across all %d analogies: %.4f (Exact); %.4f (MRR); %.0f (MR)"
        % (
            all_tested,
            all_exact / all_tested,
            all_correct / all_tested,
            all_tested / all_correct if all_correct > 0 else float("inf"),
        )
    )
    for t in t_tested:
        print(
            '...Analogy performance across %d "%s" relation types: %.4f (Exact); %.4f (MRR); %.0f (MR)'
            % (
                t_tested[t],
                t,
                t_exact[t] / t_tested[t],
                t_correct[t] / t_tested[t],
                t_tested[t] / t_correct[t] if t_correct[t] > 0 else float("inf"),
            )
        )
        print("\trelation\tN\texact\tMRR\tMR")
        for (_t, r) in r_tested:
            if _t == t:
                print(
                    "\t%s\t%d\t%.4f\t%.4f\t%.0f"
                    % (
                        r,
                        r_tested[(t, r)],
                        r_exact[(t, r)] / r_tested[(t, r)],
                        r_correct[(t, r)] / r_tested[(t, r)],
                        r_tested[(t, r)] / r_correct[(t, r)]
                        if r_correct[(t, r)] > 0
                        else float("inf"),
                    )
                )
