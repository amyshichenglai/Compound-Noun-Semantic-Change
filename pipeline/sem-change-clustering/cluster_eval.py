from collections import defaultdict, Counter
import functools
from typing import Union, List, Tuple, Optional, Dict, Any, Sequence, Literal

from munkres import Munkres
import numpy as np
import scipy
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
import sklearn

from data import Sentence, CorpusData, MAX_SEQ_LENGTH, tf_idf_features
from test_items_IO import (
    TestCompounds, CompositionalityRating,
    CompositionalityRatingCordeiro, CompositionalityRatingGhost,
    TestWords, TestWord,
    RelatedCompoundSet,
)
from util import year_to_coarse_slice, COARSE_TIME_SLICES

DEBUG = False

BY_ERA = "BY_ERA"
BY_TARGET = "BY_TARGET"
BY_TARGET_EARLY_ONLY = "BY_TARGET_EARLY"
BY_TARGET_LATE_ONLY = "BY_TARGET_LATE"

class ClusterItem:
    """Not to get too aggressively OOP around here, but since this is a site
       of *many research decisions*, it makes sense to try to insulate it from
       other code a little bit. """
    weight_contextual = 1.0
    weight_random_index = 1.0
    weight_frequency = 1.0
    def __init__(self,
                 *,
                 cluster_item_id: str,
                 associated_target: str,
                 associated_sent: Sentence,
                 lemma_span: Tuple[int, int],
                 contextual_embedding: Optional["ClusterFeatureEmbedding"]=None,
                 random_indexing_embedding: Optional["ClusterFeatureEmbedding"]=None,
                 second_order_random_indexing_embedding: Optional["ClusterFeatureEmbedding"]=None,
                 frequency: Optional["ClusterFeatureFrequency"]=None,
    ):
        self.id = cluster_item_id
        self.target = associated_target
        self.sent = associated_sent
        self.lemma_span = lemma_span
        self.contextual_embedding: Optional["ClusterFeatureEmbedding"] = contextual_embedding
        self.random_indexing_embedding: Optional["ClusterFeatureEmbedding"] = random_indexing_embedding
        self.second_order_random_indexing_embedding: Optional["ClusterFeatureEmbedding"] = second_order_random_indexing_embedding
        self.frequency: Optional["ClusterFeatureFrequency"] = frequency


    def __str__(self):
        return self.sent.to_tokens()

    def str_with_span_delimited(self):
        tokens_list = [t for t in self.sent.tokens]
        tokens_list.insert(self.lemma_span[0], "☛☛ ")
        tokens_list.insert(self.lemma_span[1] + 1, " ☚☚")
        return " ".join(t for t in tokens_list)

    def __repr__(self):
        return self.sent.to_tokens()

    def distance(self, other: "ClusterItem"):
        """
        compare the various features pairwise between self and other,
        return a distance, having weighted the contribution of each
        separate feature-wise distance
        """
        dist_ce, dist_rie, dist_f, dist_second = 0.0, 0.0, 0.0, 0.0
        if self.contextual_embedding:
            if other.contextual_embedding:
                dist_ce = self.contextual_embedding.distance(other.contextual_embedding)
        if self.random_indexing_embedding:
            if other.random_indexing_embedding:
                dist_rie = self.random_indexing_embedding.distance(other.random_indexing_embedding)
        if self.second_order_random_indexing_embedding:
            if other.second_order_random_indexing_embedding:
                dist_second = self.second_order_random_indexing_embedding.distance(other.second_order_random_indexing_embedding)
        if self.frequency:
            if other.frequency and other.target == self.target:
                dist_f = self.frequency.distance(other.frequency)
        return sum([
            dist_ce * ClusterItem.weight_contextual,
            dist_rie * ClusterItem.weight_random_index,
            dist_f * ClusterItem.weight_frequency,
            dist_second * ClusterItem.weight_random_index,
        ])

    @property
    def concatenated(self):
        # TODO: the much more efficient way of dealing with this is to figure out how long the
        #       total array size is, and allocate that array one time, then copy everything into it
        vec = np.array([])
        if self.contextual_embedding:
            vec = np.concatenate([vec, self.contextual_embedding.avg_embedding])
        if self.random_indexing_embedding:
            vec = np.concatenate([vec, self.random_indexing_embedding.avg_embedding])
        if self.second_order_random_indexing_embedding:
            vec = np.concatenate([vec, self.second_order_random_indexing_embedding.avg_embedding])
        # if not self.frequency:
        #     self.frequency = ClusterFeatureFrequency(0.0000001) # could use sys.float_info.epsilon
        if self.frequency:
            vec = np.concatenate([vec, np.array([self.frequency.numerator / self.frequency.frequency])])

        return vec

    @property
    def normalized(self):
        # (needs to move from (n,) to (1, n) for normalization)
        concat = self.concatenated.reshape(1, -1)
        normalized = sklearn.preprocessing.normalize(concat)
        return normalized.squeeze()

    @property
    def weighted(self):
        # (needs to move from (n,) to (1, n) for normalization)
        concat = self.concatenated.reshape(1, -1)
        normalized = sklearn.preprocessing.normalize(concat)
        return np.multiply(normalized, self.weight_vector).squeeze()

    @property
    def weight_vector(self):
        weight_vec = np.array([])
        if self.contextual_embedding:
            weight_vec = np.concatenate([weight_vec, np.array([ClusterItem.weight_contextual] * self.contextual_embedding.avg_embedding.size)])
        if self.random_indexing_embedding:
            weight_vec = np.concatenate([weight_vec, np.array([ClusterItem.weight_random_index] * self.random_indexing_embedding.avg_embedding.size)])
        if self.second_order_random_indexing_embedding:
            weight_vec = np.concatenate([weight_vec, np.array([ClusterItem.weight_random_index] * self.second_order_random_indexing_embedding.avg_embedding.size)])
        if self.frequency:
            weight_vec = np.concatenate([weight_vec, np.array([ClusterItem.weight_frequency])])
        return weight_vec


    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def target_tokens_char_span(self) -> Tuple[int, int]:
        """get char span of target within the target sentence
        (viewed as TOKENS not as LEMMAS)"""
        char_start = len(" ".join(t for t in self.sent.tokens[:self.lemma_span[0]]))
        char_end = char_start + len(" ".join(t for t in self.sent.tokens[self.lemma_span[0]: self.lemma_span[1]]))
        return char_start, char_end

def distance_vectorized(reference_obj: ClusterItem, concatenated_1, concatenated_2):
        assert len(concatenated_1) == len(concatenated_2)
        range_ce = (0, len(reference_obj.contextual_embedding.avg_embedding)) \
            if reference_obj.contextual_embedding else (0, 0)
        range_rie = (range_ce[1], range_ce[1] + len(reference_obj.random_indexing_embedding.avg_embedding)) \
            if reference_obj.random_indexing_embedding else (range_ce[1], range_ce[1])
        range_second = (range_rie[1], range_rie[1] + len(reference_obj.second_order_random_indexing_embedding.avg_embedding)) \
            if reference_obj.second_order_random_indexing_embedding else (range_rie[1], range_rie[1])
        range_f = (range_second[1], range_second[1] + 1) if reference_obj.second_order_random_indexing_embedding else (range_second[1], range_second[1])
        dist_ce, dist_rie, dist_second, dist_f = 0.0, 0.0, 0.0, 0.0
        dist_ce = scipy.spatial.distance.cosine(
            concatenated_1[range_ce[0]: range_ce[1]],
            concatenated_2[range_ce[0]: range_ce[1]]
        )
        dist_rie = scipy.spatial.distance.cosine(
            concatenated_1[range_rie[0]: range_rie[1]],
            concatenated_2[range_rie[0]: range_rie[1]],
        )
        dist_second = scipy.spatial.distance.cosine(
            concatenated_1[range_second[0]: range_second[1]],
            concatenated_2[range_second[0]: range_second[1]],
        )
        dist_f = abs(ClusterFeatureFrequency.numerator / concatenated_1[range_f[0]: range_f[1]] - \
                     ClusterFeatureFrequency.numerator / concatenated_2[range_f[0]: range_f[1]])
        return sum([
            dist_ce * ClusterItem.weight_contextual,
            dist_rie * ClusterItem.weight_random_index,
            dist_second * ClusterItem.weight_random_index,
            dist_f * ClusterItem.weight_frequency
        ])

class ClusterFeatureEmbedding:
    """Anything that's being compared via cosine distance"""
    def __init__(self, embeddings: Union[List[np.ndarray], np.ndarray]):
        if type(embeddings) == np.ndarray:
            self._embedding = embeddings
        else:
            self._embedding = None
        self.embeddings = embeddings

    def distance(self, other: "ClusterFeatureEmbedding"):
        return scipy.spatial.distance.cosine(self.avg_embedding, other.avg_embedding)

    @property
    def avg_embedding(self):
        if self._embedding is None:
            if len(self.embeddings) == 1:
                self._embedding = self.embeddings[0]
            else:
                self._embedding = functools.reduce(
                    lambda acc, x: acc.__add__(x), self.embeddings, np.zeros_like(self.embeddings[0])
                ) / len(self.embeddings)
        return self._embedding

class ClusterFeatureCompoundStats:
    def __init__(self, timeslice, frequency, productivity):
        self.timeslice: Tuple[int, int] = timeslice
        self.frequency = frequency
        self.productivity = productivity
        # probably frequency should have already been normalized by the size of the slice

    def distance(self, other: "ClusterFeatureCompoundStats"):
        # OR we could use these as two dimensions and deal with them separately
        # alternative would be squared error (would emphasize larger differences)
        # i.e.: return (self.frequency - other.frequency) ** 2 + (self.productivity - other.productivity) ** 2
        return abs(self.frequency - other.frequency) + abs(self.productivity - other.productivity)

class ClusterFeatureFrequency:
    numerator = 4
    def __init__(self, frequency):
        self.frequency = frequency
        assert self.frequency > 0

    def distance(self, other: "ClusterFeatureFrequency"):
        # avoid div by zero

        return abs(ClusterFeatureFrequency.numerator / self.frequency - ClusterFeatureFrequency.numerator / other.frequency)

class SupervisedTestItem:
    def __init__(self, text: str, rating: Union[bool, float]):
        self.text = text
        self.rating = rating # bool for binary, float for graded

    @classmethod
    def from_input(cls, input_line):
        target, rating = input_line.strip().split('\t')
        if "_" in target:
            target = target.split("_")[0] # remove _nn marker
        if rating in ['0', '1']:
            rating = bool(int(rating))
        else:
            rating = float(rating)
        return cls(target, rating)

class UnsupervisedClusterEval:
    def __init__(
            self,
            method: Literal['silhouette', 'davies_bouldin', 'calinski_harabasz']
    ):
        self.method = method

    def evaluate_cluster_results(self, arr, labels) -> float:
        if self.method == 'silhouette':
            return sklearn.metrics.silhouette_score(arr, labels)
        elif self.method == 'davies_bouldin':
            return sklearn.metrics.davies_bouldin_score(arr, labels)
        elif self.method == 'calinski_harabasz':
            return sklearn.metrics.calinski_harabasz_score(arr, labels)
        raise ValueError(f"unknown cluster eval method: {self.method}")


def eval_clusters(
        items: List[ClusterItem],
        labels: np.ndarray,
        gold: SupervisedTestItem,
        single_target_clustering: bool,
        era_1: Tuple[int, int],
        era_2: Optional[Tuple[int, int]]=None,
) -> Union[bool, float]:
    """

    """

    # need to take distribution (per target) of cluster items into clusters...
    # and depending on how these correspond w/ the eras... make a binary
    # decision about whether there was any change...
    cluster_to_item = defaultdict(list)
    # (to do cluster to item we would need to make ClusterItems hashable)
    for i, item in enumerate(items):
        cluster_to_item[labels[i]].append(item)

    # what is the likelihood that an item of time t1 belongs to cluster n?

    # should temporal belonging be binary or graded (distance to boundary between t1 and t2?)

    # if there is a cluster distinctive for t1 or t2 (conditional prob?), then we should output true
    # what's the threshold there???

    ##*** this all depends on the granularity of "what to cluster"

def eval_clusters_multi_target(
        params,
        examples: List[ClusterItem],
        cluster_labels: np.ndarray,
        gold: Optional[Dict[str, SupervisedTestItem]],
        threshold_clusters: float,
        logging=None,
):
    era_1 = COARSE_TIME_SLICES[params['dataset_type']][0]
    era_2 = COARSE_TIME_SLICES[params['dataset_type']][-1]
    # first condition on the target,
    # *then* on the time period
    items_by_target_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_target_by_time = defaultdict(lambda: defaultdict(list))
    items_by_cluster = defaultdict(list)
    for i, cluster_item in enumerate(examples):
        year = cluster_item.sent.year
        lab = cluster_labels[i]
        if year_to_coarse_slice(year, params['dataset_type']) == era_1:
            items_by_target_by_time[cluster_item.target][str(era_1)].append(cluster_item)
            cluster_labels_by_target_by_time[cluster_item.target][str(era_1)].append(lab)
        elif year_to_coarse_slice(year, params['dataset_type']) == era_2:
            items_by_target_by_time[cluster_item.target][str(era_2)].append(cluster_item)
            cluster_labels_by_target_by_time[cluster_item.target][str(era_2)].append(lab)
        items_by_cluster[lab].append(cluster_item)
    outputs = {}
    for target in items_by_target_by_time:
        if len(items_by_target_by_time[target]) < 2:
            if logging:
                logging.info(f"{target} doesn't occur in both time periods")
            continue
        try:
            div, t1_dist, t2_dist = calc_divergence(
                cluster_labels,
                items_by_target_by_time[target],
                cluster_labels_by_target_by_time[target],
                threshold_clusters
            )
            outputs[str(target)] = {"div_t1_t2": div}
        except RuntimeError:
            if logging:
                logging.info(f"could not cluster {target} as no clusters remained after"
                             f" thresholding")
    return outputs

def eval_clusters_multi_target_with_respect_to_constituents(
    params,
    examples: List[ClusterItem],
    cluster_labels: np.ndarray,
    gold: Optional[Dict[str, SupervisedTestItem]],
    threshold_clusters: float,
    target_compounds: List[CompositionalityRating],
    constituent_type: Literal["mod", "head"],
    logging=None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """

    :param examples:
    :param cluster_labels:
    :param gold:
    :param era_1:
    :param era_2:
    :param threshold_clusters:
    :param target_compounds: list of compound, mod, head groupings,
                             used to eval with respect to constituent
    :param constituent_types:
    :return:
    """
    era_1 = COARSE_TIME_SLICES[params['dataset_type']][0]
    era_2 = COARSE_TIME_SLICES[params['dataset_type']][-1]
    # first condition on the target,
    # *then* on the time period
    compound_targets = [c.compound for c in target_compounds]
    constituent_targets = [getattr(c, constituent_type) for c in target_compounds]
    items_by_target_compound_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_target_compound_by_time = defaultdict(lambda: defaultdict(list))
    items_by_target_constituent_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_target_constituent_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_time = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_early = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_late = defaultdict(lambda: defaultdict(int))
    items_by_cluster = defaultdict(list)
    for i, cluster_item in enumerate(examples):
        year = cluster_item.sent.year
        lab = cluster_labels[i]
        cluster_labels_by_str[str(int(lab))][cluster_item.target] += 1
        if year_to_coarse_slice(year, params['dataset_type']) == era_1:
            cluster_labels_by_time[str(int(lab))][str(era_1)] += 1
            cluster_labels_by_str_early[str(int(lab))][cluster_item.target] += 1
            if cluster_item.target in compound_targets:
                items_by_target_compound_by_time[cluster_item.target][str(era_1)].append(cluster_item)
                cluster_labels_by_target_compound_by_time[cluster_item.target][str(era_1)].append(lab)
            elif cluster_item.target in constituent_targets:
                items_by_target_constituent_by_time[cluster_item.target][str(era_1)].append(cluster_item)
                cluster_labels_by_target_constituent_by_time[cluster_item.target][str(era_1)].append(lab)
        elif year_to_coarse_slice(year, params['dataset_type']) == era_2:
            cluster_labels_by_time[str(int(lab))][str(era_2)] += 1
            cluster_labels_by_str_late[str(int(lab))][cluster_item.target] += 1
            if cluster_item.target in compound_targets:
                items_by_target_compound_by_time[cluster_item.target][str(era_2)].append(cluster_item)
                cluster_labels_by_target_compound_by_time[cluster_item.target][str(era_2)].append(lab)
            elif cluster_item.target in constituent_targets:
                items_by_target_constituent_by_time[cluster_item.target][str(era_2)].append(cluster_item)
                cluster_labels_by_target_constituent_by_time[cluster_item.target][str(era_2)].append(lab)
        items_by_cluster[lab].append(cluster_item)



    outputs = {} # keyed by target
    for target_compound, target_constituent in zip(compound_targets, constituent_targets):
        if len(items_by_target_compound_by_time[target_compound]) == 2:
            try:
                compound_single_div, dist_t1, dist_t2 = calc_divergence(
                    labels=cluster_labels,
                    items_by_time=items_by_target_compound_by_time[target_compound],
                    labels_by_time=cluster_labels_by_target_compound_by_time[target_compound],
                    threshold=threshold_clusters,
                )
                if str(target_compound) not in outputs:
                    outputs[str(target_compound)] = {}
                outputs[str(target_compound)]["div_t1_t2"] = compound_single_div
            except RuntimeError:
                if logging:
                    logging.warning(f"{target_compound} could not be evaluated across t1 / t2")
        if len(items_by_target_constituent_by_time[target_constituent]) == 2:
            try:
                constit_single_div, dist_t1, dist_t2 = calc_divergence(
                    labels=cluster_labels,
                    items_by_time=items_by_target_constituent_by_time[target_constituent],
                    labels_by_time=cluster_labels_by_target_constituent_by_time[target_constituent],
                    threshold=threshold_clusters,
                )
                if str(target_constituent) not in outputs:
                    outputs[str(target_constituent)] = {}
                outputs[str(target_constituent)]["div_t1_t2"] = constit_single_div
            except RuntimeError:
                if logging:
                    logging.warning(f"{target_constituent} could not be evaluated across t1 / t2")
        # dealing with both the target compound and the constituent together:
        if len(items_by_target_compound_by_time[target_compound]) < 2 or \
                len(items_by_target_constituent_by_time[target_constituent]) < 2:
            if logging:
                logging.info(f"{target_compound} or {target_constituent} not in both time periods: {len(items_by_target_compound_by_time[target_compound])}, {len(items_by_target_constituent_by_time[target_constituent])}")
            outputs[str(target_compound)] = None
            continue
        try:
            divergences, compound_distributions, constituent_distributions = calc_divergence_for_two_targets(
                labels=cluster_labels,
                items1_by_time=items_by_target_compound_by_time[target_compound],
                labels1_by_time=cluster_labels_by_target_compound_by_time[target_compound],
                items2_by_time=items_by_target_constituent_by_time[target_constituent],
                labels2_by_time=cluster_labels_by_target_constituent_by_time[target_constituent],
                threshold=threshold_clusters,
                logging=logging,
            )
            if target_compound not in outputs:
                outputs[target_compound] = {}
            outputs[str(target_compound)].update({
                "div_t1": divergences[0], "div_t2": divergences[1],
            })

            if DEBUG:
                outputs[str(target_compound)] = outputs[str(target_compound)] | {"misc": {
                    "compound_dist_t1": compound_distributions[0],
                    "compound_dist_t2": compound_distributions[1],
                    "constituent_dist_t1": constituent_distributions[0],
                    "constituent_dist_t2": constituent_distributions[1],
                    }
                }
        except RuntimeError:
            if logging:
                logging.info(f"{target_compound}, {target_constituent} could not be evaluated, "
                             f" no clusters remained after threshold cutoff")



    # also include some global level output:
    global_outputs = {}
    # items_by_cluster: cluster label to list of ClusterItem
    global_outputs[BY_ERA] = dict(cluster_labels_by_time)
    global_outputs[BY_TARGET] = dict(cluster_labels_by_str)
    global_outputs[BY_TARGET_EARLY_ONLY] = dict(cluster_labels_by_str_early)
    global_outputs[BY_TARGET_LATE_ONLY] = dict(cluster_labels_by_str_late)

    return outputs, global_outputs

def eval_clusters_multi_target_with_shared_constituent_groups(
    params,
    examples: List[ClusterItem],
    cluster_labels: np.ndarray,
    gold,
    threshold_clusters: float,
    target_compounds: List[RelatedCompoundSet],
    logging=None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # 2 evals:
    # 1.) per time slice pairwise JSD between 'primary component' compound and all others
    # 2.) JSD (t1, t2) for each compound in each set -- and average across these
    primary_compound_targets = [c.primary_component for c in target_compounds]
    related_compound_sets = [c.secondary_components for c in target_compounds]
    all_related_compounds = [related for compound_set in related_compound_sets for related in compound_set]

    items_by_target_compound_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_target_compound_by_time = defaultdict(lambda: defaultdict(list))
    items_by_related_compound_by_time = defaultdict(lambda: defaultdict(list))
    cluster_labels_by_related_compound_by_time = defaultdict(lambda: defaultdict(list))

    cluster_labels_by_time = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_early = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_late = defaultdict(lambda: defaultdict(int))
    items_by_cluster = defaultdict(list)

    era_1 = COARSE_TIME_SLICES[params['dataset_type']][0]
    era_2 = COARSE_TIME_SLICES[params['dataset_type']][-1]

    for i, cluster_item in enumerate(examples):
        year = cluster_item.sent.year
        lab = cluster_labels[i]
        cluster_labels_by_str[str(int(lab))][cluster_item.target] += 1
        if year_to_coarse_slice(year, params['dataset_type']) == era_1:
            cluster_labels_by_time[str(int(lab))][str(era_1)] += 1
            cluster_labels_by_str_early[str(int(lab))][cluster_item.target] += 1
            if cluster_item.target in primary_compound_targets:
                items_by_target_compound_by_time[cluster_item.target][str(era_1)].append(cluster_item)
                cluster_labels_by_target_compound_by_time[cluster_item.target][str(era_1)].append(lab)
            if cluster_item.target in all_related_compounds:
                items_by_related_compound_by_time[cluster_item.target][str(era_1)].append(cluster_item)
                cluster_labels_by_related_compound_by_time[cluster_item.target][str(era_1)].append(lab)
        elif year_to_coarse_slice(year, params['dataset_type']) == era_2:
            cluster_labels_by_time[str(int(lab))][str(era_2)] += 1
            cluster_labels_by_str_late[str(int(lab))][cluster_item.target] += 1
            if cluster_item.target in primary_compound_targets:
                items_by_target_compound_by_time[cluster_item.target][str(era_2)].append(cluster_item)
                cluster_labels_by_target_compound_by_time[cluster_item.target][str(era_2)].append(lab)
            if cluster_item.target in all_related_compounds:
                items_by_related_compound_by_time[cluster_item.target][str(era_2)].append(cluster_item)
                cluster_labels_by_related_compound_by_time[cluster_item.target][str(era_2)].append(lab)
        items_by_cluster[lab].append(cluster_item)

    # ALL targets (primary or secondary) are used as keys, with
    # sub-entries under "div_t1_t2"
    # PRIMARY targets get sub entries nested under the keys of their
    # related compounds, under which will be "div_t1" and "div_t2" entries
    outputs = {} # keyed by target
    for target_compound, related_compounds_list in zip(primary_compound_targets, related_compound_sets):
        pairwise_div_t1, pairwise_div_t2 = [], []
        if str(target_compound) not in outputs:
            outputs[str(target_compound)] = {}
        if len(items_by_target_compound_by_time[target_compound]) < 2:
            if logging:
                logging.info(f"{target_compound} (target compound) not in both time periods:"
                             f" {len(items_by_target_compound_by_time[target_compound])}")
            # outputs[str(target_compound)] = None
        elif len(items_by_target_compound_by_time[target_compound]) == 2:
            try:
                target_compound_single_div, dist_t1, dist_t2 = calc_divergence(
                    labels=cluster_labels,
                    items_by_time=items_by_target_compound_by_time[target_compound],
                    labels_by_time=cluster_labels_by_target_compound_by_time[target_compound],
                    threshold=threshold_clusters,
                )
                outputs[str(target_compound)]["div_t1_t2"] = target_compound_single_div
            except RuntimeError:
                if logging:
                    logging.warning(f"{target_compound} could not be evaluated across t1 / t2")
        for related_compound in related_compounds_list:
            if len(items_by_related_compound_by_time[related_compound]) < 2:
                if logging:
                    logging.info(f"{related_compound} (related compound) not in both time periods:"
                                 f"{len(items_by_related_compound_by_time[related_compound])}")
                outputs[str(related_compound)] = None
                continue
                # last but not least we run the target compounds through the t1/t2 divergence
            try:
                related_compound_single_div, dist_t1, dist_t2 = calc_divergence(
                    labels=cluster_labels,
                    items_by_time=items_by_related_compound_by_time[related_compound],
                    labels_by_time=cluster_labels_by_related_compound_by_time[related_compound],
                    threshold=threshold_clusters,
                )
                if str(related_compound) not in outputs:
                    outputs[str(related_compound)] = {}
                outputs[str(related_compound)]["div_t1_t2"] = related_compound_single_div
            except RuntimeError:
                if logging:
                    logging.warning(f"{related_compound} could not be evaluated across t1 / t2")

            # can skip the analysis of both together
            if len(items_by_target_compound_by_time[target_compound]) < 2:
                continue

            try:
                divergences, compound_distributions, constituent_distributions = calc_divergence_for_two_targets(
                    labels=cluster_labels,
                    items1_by_time=items_by_target_compound_by_time[target_compound],
                    labels1_by_time=cluster_labels_by_target_compound_by_time[target_compound],
                    items2_by_time=items_by_related_compound_by_time[related_compound],
                    labels2_by_time=cluster_labels_by_related_compound_by_time[related_compound],
                    threshold=threshold_clusters,
                    logging=logging,
                )
                outputs[str(target_compound)][str(related_compound)] = {
                    "div_t1": divergences[0], "div_t2": divergences[1],
                }
                pairwise_div_t1.append(divergences[0])
                pairwise_div_t2.append(divergences[1])
            except RuntimeError:
                if logging:
                    logging.info(f"{target_compound}, {related_compound} could not be evaluated, "
                                 f" no clusters remained after threshold cutoff")

        # for this target compound, we calc the average pairwise divergence for its related componuds
        outputs[str(target_compound)]["avg_pairwise_div_t1"] = \
            sum(pairwise_div_t1) / len(pairwise_div_t1) if len(pairwise_div_t1) else 0
        outputs[str(target_compound)]["avg_pairwise_div_t2"] = \
            sum(pairwise_div_t2) / len(pairwise_div_t2) if len(pairwise_div_t2) else 0


    # also include some global level output:
    global_outputs = {}
    # items_by_cluster: cluster label to list of ClusterItem
    global_outputs[BY_ERA] = dict(cluster_labels_by_time)
    global_outputs[BY_TARGET] = dict(cluster_labels_by_str)
    global_outputs[BY_TARGET_EARLY_ONLY] = dict(cluster_labels_by_str_early)
    global_outputs[BY_TARGET_LATE_ONLY] = dict(cluster_labels_by_str_late)

    return outputs, global_outputs


def eval_clusters_all_eras_single_target(
        params,
        examples: List[ClusterItem],
        cluster_labels: np.ndarray,
        gold: Optional[SupervisedTestItem],
        threshold_clusters: float,
        # threshold_change: float,
        logging=None,
) -> Tuple[float, Tuple[int, int], Dict[str, Any]]:
    # what is the relationship between a cluster & time periods
    items_by_time = defaultdict(list)
    items_by_cluster = defaultdict(list)
    labels_by_time = defaultdict(list)

    cluster_labels_by_time = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_early = defaultdict(lambda: defaultdict(int))
    cluster_labels_by_str_late = defaultdict(lambda: defaultdict(int))

    era_1 = COARSE_TIME_SLICES[params['dataset_type']][0]
    era_2 = COARSE_TIME_SLICES[params['dataset_type']][-1]

    for i in range(len(examples)):
        it = examples[i]
        lab = cluster_labels[i]
        year = it.sent.year
        if year_to_coarse_slice(year, params['dataset_type']) == era_1:
            items_by_time[str(era_1)].append(it)
            labels_by_time[str(era_1)].append(lab)
            cluster_labels_by_time[str(int(lab))][str(era_1)] += 1
            cluster_labels_by_str_early[str(int(lab))][it.target] += 1
        elif year_to_coarse_slice(year, params['dataset_type']) == era_2:
            items_by_time[str(era_2)].append(it)
            labels_by_time[str(era_2)].append(lab)
            cluster_labels_by_time[str(int(lab))][str(era_2)] += 1
            cluster_labels_by_str_late[str(int(lab))][it.target] += 1
        items_by_cluster[lab].append(it)
        cluster_labels_by_str_early[str(int(lab))][it.target] += 1

    if len(items_by_time) < 2:
        print(f"items only available in one time slice")
        return None, None, None, None
    div, t1_dist, t2_dist = calc_divergence(
        cluster_labels, items_by_time, labels_by_time, threshold_clusters)
    gains, losses = single_target_gain_or_loss(
        labels_by_time[str(era_1)],
        labels_by_time[str(era_2)],
        min_total_count=10,
        min_stable_count=2,
    )
    misc_output = {"t1_dist": t1_dist, "t2_dist": t2_dist}
    prediction_type = type(gold.rating) if gold is not None else type(bool)

    # also include some global level output:
    misc_output[BY_ERA] = dict(cluster_labels_by_time)
    misc_output[BY_TARGET] = dict(cluster_labels_by_str)
    misc_output[BY_TARGET_EARLY_ONLY] = dict(cluster_labels_by_str_early)
    misc_output[BY_TARGET_LATE_ONLY] = dict(cluster_labels_by_str_late)

    return div, (gains, losses), misc_output

    # TODO: use supervised label against label determined by
    #       local calculation


def map_clusters(
        params,
        all_items: List[ClusterItem],
        labels_1: np.ndarray,
        labels_2: np.ndarray,
) -> Sequence[Tuple[int, int]]:
    """Create a mapping between two sets of clusters
    (which use the same feature space but non-overlappping items).
    Similarity between medoid items per cluster
    """
    time_1 = COARSE_TIME_SLICES[params['dataset_type']][0]
    time_2 = COARSE_TIME_SLICES[params['dataset_type']][-1]

    items_1 = [it for it in all_items if year_to_coarse_slice(it.sent.year, params['dataset_type']) == time_1]
    items_2 = [it for it in all_items if year_to_coarse_slice(it.sent.year, params['dataset_type']) == time_2]
    all_labels_1 = sorted(list(set([k for k in labels_1])))
    all_labels_2 = sorted(list(set([k for k in labels_2])))
    all_labels = all_labels_1 + all_labels_2
    items_by_cluster_1 = defaultdict(list)
    for i, it_1 in enumerate(items_1):
        items_by_cluster_1[labels_1[i]].append(it_1.avg_embedding)
    label_to_medoid_1 = {}
    for label in all_labels_1:
        # pairwise dist of items in cluster:
        dist_m = np.ndarray((len(items_by_cluster_1[label]), len(items_by_cluster_1[label])))
        for i in range(len(items_by_cluster_1[label])):
            for j in range(len(items_by_cluster_1[label])):
                dist_m[i][j] = scipy.spatial.distance.cosine(items_by_cluster_1[label][i], items_by_cluster_1[label][j])
        # take argmin of row or col sum of the cluster pairwise dist matrix to find the
        # cluster item closest to all others, have it stand in for the cluster label:
        label_to_medoid_1[label] = items_by_cluster_1[label][np.argmin(dist_m.sum(axis=0))]
    items_by_cluster_2 = defaultdict(list)
    for i, it_2 in enumerate(items_2):
        items_by_cluster_2[labels_2[i]].append(it_2.avg_embedding)
    label_to_medoid_2 = {}
    for label in all_labels_2:
        dist_m = np.ndarray((len(items_by_cluster_2[label]), len(items_by_cluster_2[label])))
        for i in range(len(items_by_cluster_2[label])):
            for j in range(len(items_by_cluster_2[label])):
                dist_m[i][j] = scipy.spatial.distance.cosine(items_by_cluster_2[label][i], items_by_cluster_2[label][j])
        label_to_medoid_2[label] = items_by_cluster_2[label][np.argmin(dist_m.sum(axis=0))]

    # we need to make this a square matrix, even if there are different numbers of clusters
    # because that's what the munkres library expects
    matrix_cardinality = max(len(all_labels_1), len(all_labels_2))
    diff_matrix = np.zeros((matrix_cardinality, matrix_cardinality))
    for i, lab_1 in enumerate(all_labels_1):
        for j, lab_2 in enumerate(all_labels_2):
            diff_matrix[i][j] = scipy.spatial.distance.cosine(
                label_to_medoid_1[lab_1],
                label_to_medoid_2[lab_2]
            )
    munkres_obj = Munkres()
    return munkres_obj.compute(diff_matrix)


def mapped_cluster_divergence(
        labels_1: np.ndarray,
        labels_2: np.ndarray,
        cluster_mapping: Sequence[Tuple[int, int]]
):
    """
    The general intention here is that labels_1 come from one time period
    and labels_2 from the other. But however they're constructed,
    :param labels_1:
    :param labels_2:
    :param cluster_mapping: mappings from clustering 1 labels to clustering 2 labels
    (which can be the same int, but refer to something different!)
    :return:
    """
    # we first use the cluster_mapping to create a unified set of labels
    l1_to_unified = defaultdict(int)
    l2_to_unified = defaultdict(int)
    u_label_counter = 0
    for l1, l2 in cluster_mapping:
        l1_to_unified[l1] = u_label_counter
        l2_to_unified[l2] = u_label_counter
        u_label_counter += 1
    u_clusters_1 = [l1_to_unified[labels_1[i]] for i in range(len(labels_1))]
    u_clusters_2 = [l2_to_unified[labels_2[i]] for i in range(len(labels_2))]
    u_clusters_1_label_counts = Counter(u_clusters_1)
    u_clusters_2_label_counts = Counter(u_clusters_2)
    u1_clusters, u2_clusters = [], []
    for i in range(len(cluster_mapping)):
        # TODO: here's where we could apply a threshold
        u1_clusters.append(u_clusters_1_label_counts[i])
        u2_clusters.append(u_clusters_2_label_counts[i])
    u1_clusters = np.array(u1_clusters)
    u2_clusters = np.array(u2_clusters)

    u_clusters_1_dist = u1_clusters / u1_clusters.sum()
    u_clusters_2_dist = u2_clusters / u2_clusters.sum()
    return jsd(u_clusters_1_dist, u_clusters_2_dist), u1_clusters.tolist(), u2_clusters.tolist()

def predict_change_in_sense_inventory(
        divergence: float,
        cluster_gains: int,
        cluster_losses: int,
        difference_type: type,
        divergence_threshold: float,
):
    if difference_type is float:
        raise NotImplementedError
    else:
        if divergence > divergence_threshold and (cluster_gains > 0 or cluster_losses < 0):
            return True
        return False


def cluster_diagnostics(
        clustered_items: List[ClusterItem],
        cluster_labels: np.ndarray, # arr of integers
        targets: List[Union[CompositionalityRating, TestWord]], # there's also the SemEval targets...

):
    """Basic idea -> we have the original sentences, and their assignment into clusters.
       We want to be able to, for each/any given target, rank examples by
       their centrality in a cluster
       so... take the cluster w/ the most examples of the given target in it, and report
       the examples most central and most marginal to that cluster's centroid

       this involves some amount of duplicate work to recreate a bit of the
       clustering algo internals, where centroids and the like are calculated
       anyway.

       In the particular case of the compound x constituent clustering,
       we can try to correlate divergence scores (maybe only for the more modern era)
       against compositionality ratings.

       what should be the output format? write to log? write to a file?
    """
    print(f"running cluster diagnostics:")
    clusters = defaultdict(list)
    target_str_to_cluster_list = defaultdict(lambda: defaultdict(int))
    for i, cluster_item in enumerate(clustered_items):
        clusters[cluster_labels[i]].append(cluster_item)
        target_str_to_cluster_list[cluster_item.target][cluster_labels[i]] += 1

    # compute centroid
    centroids = []
    for cluster in clusters.values():
        centroids.append(
            np.sum([c.concatenated for c in cluster], axis=0) / len(cluster) if len(cluster) else None
        )
    for target in targets:
        # which clusters have the largest portion of the thing
        top_clusters = sorted([(cluster, count)
                               for cluster, count
                            in target_str_to_cluster_list[target.primary_component].items()],
                             key=lambda x: x[1], reverse=True) # sort on count, descending
        # from there we take the top N
        top_clusters = [cluster for cluster, count in top_clusters[:5]]

        # now in those three clusters, rank the instances of the target
        # according to distance from the centroid of the cluster
        for top_cluster_idx, cluster in enumerate(top_clusters):
            instances = [cluster_item for i, cluster_item in enumerate(clustered_items)
                     if (cluster_labels[i] == cluster and cluster_item.target == target.primary_component)]
            if not instances:
                continue
            # calc distances to centroid for this cluster
            dists = []
            for i, instance in enumerate(instances):
                dists.append(
                    (euclidean(centroids[top_cluster_idx], instances[i].concatenated, w=instances[i].weight_vector),
                     i)
                )
            dists.sort()
            # TODO: could we also calculate and display distances between compound target / constituent in
            #       this cluster? (if applicable -- they might not occur together
            print(f"===\n{target.primary_component}::cluster {top_cluster_idx} has {len(instances)} instances\n===")
            # these are tuples (dist, instance_index)
            most_central, most_peripheral = dists[:3], reversed(dists[-3:])
            print(f"Most central instances for «{target.primary_component}» top cluster {top_cluster_idx}")
            for distance, instance_idx in most_central:
                print(f"(year: {instances[instance_idx].sent.year}, dist: {distance:.02f}){instances[instance_idx].str_with_span_delimited()}\n***\n")
            print(f"Most peripheral instances for «{target.primary_component}» top cluster {top_cluster_idx}")
            for distance, instance_idx in most_peripheral:
                print(f"(year: {instances[instance_idx].sent.year}, dist: {distance:.02f}){instances[instance_idx].str_with_span_delimited()}\n***\n")

        # then output the top 5 most central, bottom 5 least central examples

    # that's pretty straightforward so far, without getting into any
    # pairwise sort of analyses hither and thither.

    # no reason not to print a raw log of everything clustered,
    # for situations where we only have ~15k or so targets in the mix
    for i in sorted([k for k in clusters.keys()]):
        print(f"---> Cluster {i} total contents:\n\n")
        for elt in clusters[i]:
            print(f"(year: {elt.sent.year}){elt.str_with_span_delimited()}\n***\n")


def calc_divergence(
        labels,
        items_by_time,
        labels_by_time,
        threshold,
        logging=None,
):
    # if items are ClusterItems, this is gonna be slow python interpreter code
    # good place to start, before converting to all numpy kind of operations
    times = sorted([k for k in items_by_time.keys()])
    t1_items = items_by_time[times[0]]
    t2_items = items_by_time[times[1]]

    t1_label_counts = Counter(labels_by_time[times[0]])
    t2_label_counts = Counter(labels_by_time[times[1]])

    all_labels = sorted(list(set([k for k in labels])))

    # apply threshold (exclude tiny clusters)
    t1_clusters, t2_clusters = [], []
    labels_above_threshold = []
    for i in all_labels:
        if t1_label_counts[i] + t2_label_counts[i] > threshold:
            t1_clusters.append(t1_label_counts[i])
            t2_clusters.append(t2_label_counts[i])
            labels_above_threshold.append(i)
    t1_clusters = np.array(t1_clusters)
    t2_clusters = np.array(t2_clusters)
    if not np.any(t1_clusters) or not np.any(t2_clusters):
        raise RuntimeError("would divide by zero to continue with empty set of clusters")
    # sort out t1 and t2 items by cluster
    # t1_items_by_cluster = defaultdict(list)
    # t2_items_by_cluster = defaultdict(list)
    # for it, lab in zip(t1_items, labels_by_time[times[0]]):
    #     if lab in labels_above_threshold:
    #         if isinstance(it, ClusterItem):
    #             t1_items_by_cluster[lab].append(it.avg_embedding)
    #         else:
    #             t1_items_by_cluster[lab].append(it[0]) # only append the array
    # for it, lab in zip(t2_items, labels_by_time[times[1]]):
    #     if lab in labels_above_threshold:
    #         if isinstance(it, ClusterItem):
    #             t2_items_by_cluster[lab].append(it.avg_embedding)
    #         else:
    #             t2_items_by_cluster[lab].append(it[0])

    # get means of cluster items for each time period (whatever this may imply!)
    # t1_per_cluster_means = np.array([cluster_mean(t1_items_by_cluster[c]) for c in labels_above_threshold])
    # t2_per_cluster_means = np.array([cluster_mean(t2_items_by_cluster[c]) for c in labels_above_threshold])



    t1_dist = t1_clusters / t1_clusters.sum() # prob dist of labels for t1
    t2_dist = t2_clusters / t2_clusters.sum()

    if logging and DEBUG:
        logging.info(f"counts: t1: {sum(t1_clusters)}, "
                     f"t2: {sum(t2_clusters)}\n\n"
                     f"Distributions:\nt1: {t1_dist}\n\nt2: {t2_dist}")

    return jsd(t1_dist, t2_dist), t1_dist.tolist(), t2_dist.tolist()

def calc_divergence_for_two_targets(
        *,
        labels,
        items1_by_time, labels1_by_time,
        items2_by_time, labels2_by_time,
        threshold,
        logging,
):
    """
    The idea here is to take two target items (clustered together) and
    compute a divergence score for the distribution of each of these item kinds
    (items1 and items2) for each of the two time periods. So we end up with
    two divergence values (one for each time period)
    :param labels:
    :param items1_by_time:
    :param labels1_by_time:
    :param items2_by_time:
    :param labels2_by_time:
    :param threshold:
    :return: Three 2-tuples:
            1.) (target1 and target2 divergences for t1,t2),
            2.) (target2 cluster distributions),
            3.) (target2 cluster distributions)
    """
    times = sorted([k for k in items1_by_time.keys()])
    t1_items1 = items1_by_time[times[0]]
    t2_items1 = items1_by_time[times[1]]
    t1_items2 = items2_by_time[times[0]]
    t2_items2 = items2_by_time[times[1]]

    t1_items1_label_counts = Counter(labels1_by_time[times[0]])
    t1_items2_label_counts = Counter(labels2_by_time[times[0]])
    t2_items1_label_counts = Counter(labels1_by_time[times[1]])
    t2_items2_label_counts = Counter(labels2_by_time[times[1]])

    all_labels = sorted(list(set([k for k in labels])))

    # apply threshold (exclude tiny clusters)
    t1_items1_clusters, t1_items2_clusters = [], []
    t2_items1_clusters, t2_items2_clusters = [], []
    labels_above_threshold = []
    for i in all_labels:
        if t1_items1_label_counts[i] + t1_items2_label_counts[i] > threshold:
            t1_items1_clusters.append(t1_items1_label_counts[i])
            t1_items2_clusters.append(t1_items2_label_counts[i])
            labels_above_threshold.append(i)
        if t2_items1_label_counts[i] + t2_items2_label_counts[i] > threshold:
            t2_items1_clusters.append(t2_items1_label_counts[i])
            t2_items2_clusters.append(t2_items2_label_counts[i])
            labels_above_threshold.append(i)
    labels_above_threshold = sorted(list(set(labels_above_threshold)))
    if logging:
        logging.info(f"counts: t1_1: {sum(t1_items1_clusters)}, "
                     f"t1_2: {sum(t1_items2_clusters)}\n"
                     f"t2_1: {sum(t2_items1_clusters)}, "
                     f"t2_2: {sum(t2_items2_clusters)}")

    t1_div, t1_items1_dist, t1_items2_dist = _two_target_divergence(
        items1=t1_items1,
        items2=t1_items2,
        items1_clusters=t1_items1_clusters,
        items2_clusters=t1_items2_clusters,
        labels1_by_time=labels1_by_time,
        labels2_by_time=labels2_by_time,
        labels_above_threshold=labels_above_threshold,
        time_period_name=times[0]
    )
    t2_div, t2_items1_dist, t2_items2_dist = _two_target_divergence(
        items1=t2_items1,
        items2=t2_items2,
        items1_clusters=t2_items1_clusters,
        items2_clusters=t2_items2_clusters,
        labels1_by_time=labels1_by_time,
        labels2_by_time=labels2_by_time,
        labels_above_threshold=labels_above_threshold,
        time_period_name=times[1],
    )
    return (t1_div, t2_div), (t1_items1_dist, t2_items1_dist), (t1_items2_dist, t2_items2_dist)


def _two_target_divergence(
    *, # it's a big mess of parameters...
    items1,
    items2,
    items1_clusters,
    items2_clusters,
    labels1_by_time,
    labels2_by_time,
    labels_above_threshold,
    time_period_name: str,
):
    items1_clusters = np.array(items1_clusters)
    items2_clusters = np.array(items2_clusters)
    if not np.any(items1_clusters) or not np.any(items2_clusters):
        raise RuntimeError(f"would divide by zero to continue")
    # reverse to get items by cluster
    items1_by_cluster = defaultdict(list)
    items2_by_cluster = defaultdict(list)
    for it, lab in zip(items1, labels1_by_time[time_period_name]):
        if lab in labels_above_threshold:
            if isinstance(it, ClusterItem):
                items1_by_cluster[lab].append(it)
            else:
                items1_by_cluster[lab].append(it[0]) # only append the array
    for it, lab in zip(items2, labels2_by_time[time_period_name]):
        if lab in labels_above_threshold:
            if isinstance(it, ClusterItem): # is this still a real distinction we need?
                items2_by_cluster[lab].append(it)
            else:
                items2_by_cluster[lab].append([it[0]])

    items1_dist = items1_clusters / items1_clusters.sum()
    items2_dist = items2_clusters / items2_clusters.sum()
    return jsd(items1_dist, items2_dist), items1_dist.tolist(), items2_dist.tolist()

def single_target_gain_or_loss(t1_labels, t2_labels, min_total_count, min_stable_count) -> Tuple[int, int]:
    """
    returns positive for gain of sense(s), negative for loss between t1 and t2
    """
    all_labels = sorted(list(set([k for k in t1_labels + t2_labels])))
    t1_counts = Counter(t1_labels)
    t2_counts = Counter(t2_labels)
    sense_gain = 0
    sense_loss = 0
    for l in all_labels:
        total_count = t1_counts[l] + t2_counts[l]
        if total_count < min_total_count:
            continue
        if l not in t1_counts or t1_counts[l] < min_stable_count:
            sense_gain += 1
        if l not in t2_counts or t2_counts[l] < min_stable_count:
            sense_loss -= 1
    return sense_gain, sense_loss


def jsd(p, q):
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2





def aggregate_score(gold_labels: List[bool], predicted_labels: List[bool], output_stream=None):
    tp, fp, fn = 0, 0, 0
    for gl, pl in zip(gold_labels, predicted_labels):
        if gl == pl:
            tp += 1
        if pl and not gl:
            fp += 1
        if gl and not pl:
            fn += 1
    precision = (
        tp / (tp + fp)
        if tp + fp != 0
        else 0
    )
    recall = (
        tp / (tp + fn)
        if tp + fn != 0
        else 0
    )
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )
    if output_stream:
        output_stream(f"tp: {tp}, fp: {fp}, fn: {fn}\n")
        output_stream(f"prec: {precision}, rec: {recall}, f1: {f1}")
    else:
        return {"aggregate": {"tp": tp, "fp": fp, "fn": fn,
                "prec": precision, "rec": recall, "f1": f1}
                }
